"""Synthetic datasets for RiskLens: credit risk (PD), card fraud, analyst decisions and an audit log.

Outputs (default: data/risklens/):
  credit_applications.csv  loan applications + customer history features, label `default`
  card_transactions.csv    card transactions with real-time-computable features, label `is_fraud`
  credit_decisions.csv     model PD score, reason codes and analyst decision for the latest applications
  fraud_alerts.csv         model fraud score, reason codes, action and analyst review for flagged transactions
  audit_log.csv            hash-chained event log of every score / explanation / human decision

Scores are produced from the generating process plus noise, standing in for a trained model,
so reason codes (top risk-increasing contributions) are consistent with the data.

Usage: python generate_risklens_data.py --seed 42 --n-applications 50000
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("data") / "risklens"

# ---------------------------------------------------------------- shared helpers


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def calibrate_intercept(z, target_rate):
    """Shift logits so that the mean probability equals target_rate (bisection)."""
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sigmoid(z + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def top_reasons(contribs: pd.DataFrame, k=3):
    """Top-k risk-increasing contributions per row as JSON [{code, weight}, ...]."""
    codes = contribs.columns.to_numpy()
    values = contribs.to_numpy()
    order = np.argsort(-values, axis=1)[:, :k]
    out = []
    for row, idx in zip(values, order):
        out.append(
            json.dumps(
                [{"code": codes[j], "weight": round(float(row[j]), 3)} for j in idx if row[j] > 0]
            )
        )
    return np.array(out)


# ---------------------------------------------------------------- credit risk

EMPLOYMENT_RISK = {"salaried": 0.0, "self_employed": 0.3, "contractor": 0.25, "unemployed": 1.2, "retired": 0.1}
PURPOSE_RISK = {
    "debt_consolidation": 0.0, "home_improvement": 0.0, "car": 0.0,
    "education": 0.05, "medical": 0.2, "business": 0.3, "other": 0.1,
}
HOME_RISK = {"own": -0.2, "mortgage": 0.0, "rent": 0.15}


def generate_credit(rng, n, default_rate):
    age = np.clip(rng.normal(40, 12, n), 21, 75).astype(int)
    income = np.clip(rng.lognormal(10.7, 0.5, n), 9_000, 400_000).round(-2)
    employment_years = np.minimum(rng.gamma(2, 3, n), age - 20).round(1)
    employment_type = rng.choice(list(EMPLOYMENT_RISK), n, p=[0.62, 0.18, 0.10, 0.04, 0.06])
    home_ownership = rng.choice(list(HOME_RISK), n, p=[0.2, 0.4, 0.4])
    purpose = rng.choice(list(PURPOSE_RISK), n, p=[0.30, 0.15, 0.15, 0.08, 0.07, 0.08, 0.17])

    credit_score = np.clip(
        rng.normal(680, 65, n) + 12 * (np.log(income) - 10.7) / 0.5 + 0.3 * (age - 40), 300, 850
    ).round().astype(int)
    loan_amount = np.clip(income * rng.uniform(0.05, 0.8, n), 1_000, 60_000).round(-2)
    term_months = rng.choice([12, 24, 36, 48, 60], n, p=[0.1, 0.2, 0.35, 0.15, 0.2])
    interest_rate = np.clip(
        5 + (750 - credit_score) / 25 + term_months / 24 + rng.normal(0, 1.2, n), 3, 35
    ).round(2)

    monthly_rate = interest_rate / 1200
    monthly_payment = loan_amount * monthly_rate / (1 - (1 + monthly_rate) ** -term_months)
    existing_debt = income / 12 * rng.beta(2, 6, n)
    dti = np.clip((monthly_payment + existing_debt) / (income / 12), 0.02, 1.2).round(3)

    credit_utilization = np.clip(rng.beta(2, 3, n) + (680 - credit_score) / 1000, 0, 1).round(3)
    delinq_lambda = 0.3 * np.clip(1 + (700 - credit_score) / 100, 0.2, 4)
    num_delinquencies_2y = rng.poisson(delinq_lambda)
    prior_defaults = (rng.random(n) < np.clip((680 - credit_score) / 1500, 0.01, 0.2)).astype(int)
    num_open_accounts = rng.poisson(6, n) + 1
    months_since_last_delinquency = np.where(num_delinquencies_2y > 0, rng.integers(1, 25, n), np.nan)
    loan_to_income = loan_amount / income

    contribs = pd.DataFrame(
        {
            "LOW_CREDIT_SCORE": -1.0 * (credit_score - 680) / 70,
            "HIGH_DTI": 0.6 * (dti - 0.35) / 0.15,
            "HIGH_UTILIZATION": 0.5 * (credit_utilization - 0.4) / 0.25,
            "RECENT_DELINQUENCIES": 0.45 * num_delinquencies_2y,
            "PRIOR_DEFAULT": 1.0 * prior_defaults,
            "LOW_INCOME": -0.4 * (np.log(income) - 10.7) / 0.5,
            "SHORT_EMPLOYMENT": -0.25 * (np.minimum(employment_years, 15) - 5) / 4,
            "EMPLOYMENT_TYPE": pd.Series(employment_type).map(EMPLOYMENT_RISK).to_numpy(),
            "HIGH_LOAN_TO_INCOME": 0.5 * (loan_to_income - 0.3) / 0.2,
            "HIGH_INTEREST_RATE": 0.3 * (interest_rate - 10) / 5,
            "LOAN_PURPOSE": pd.Series(purpose).map(PURPOSE_RISK).to_numpy(),
            "HOME_OWNERSHIP": pd.Series(home_ownership).map(HOME_RISK).to_numpy(),
        }
    )
    logit = contribs.sum(axis=1).to_numpy()
    logit = logit + calibrate_intercept(logit, default_rate)
    p_true = sigmoid(logit)
    default = (rng.random(n) < p_true).astype(int)

    start = pd.Timestamp("2024-01-01")
    seconds = np.sort(rng.integers(0, 2 * 365 * 86400, n))  # applications arrive in time order
    df = pd.DataFrame(
        {
            "application_id": [f"APP{i:07d}" for i in range(n)],
            "customer_id": [f"CUS{i:07d}" for i in rng.integers(0, int(n * 0.85), n)],
            "application_ts": start + pd.to_timedelta(seconds, unit="s"),
            "age": age,
            "annual_income": income,
            "employment_type": employment_type,
            "employment_years": employment_years,
            "home_ownership": home_ownership,
            "loan_amount": loan_amount,
            "term_months": term_months,
            "purpose": purpose,
            "interest_rate": interest_rate,
            "monthly_payment": monthly_payment.round(2),
            "dti": dti,
            "credit_score": credit_score,
            "credit_utilization": credit_utilization,
            "num_open_accounts": num_open_accounts,
            "num_delinquencies_2y": num_delinquencies_2y,
            "months_since_last_delinquency": months_since_last_delinquency,
            "prior_defaults": prior_defaults,
            "default": default,
        }
    )
    # real-world dirt, injected after the label so the signal is unaffected
    df.loc[rng.random(n) < 0.02, "employment_years"] = np.nan
    df.loc[rng.random(n) < 0.01, "credit_score"] = np.nan
    return df, contribs, logit


# ---------------------------------------------------------------- card fraud

MCC = ["grocery", "restaurant", "fuel", "travel", "electronics", "online_services", "atm_cash", "entertainment", "health", "luxury"]
MCC_P = [0.24, 0.17, 0.10, 0.05, 0.06, 0.12, 0.06, 0.08, 0.08, 0.04]
FRAUD_MCC = ["electronics", "luxury", "online_services", "travel", "atm_cash"]
FRAUD_MCC_P = [0.30, 0.20, 0.25, 0.10, 0.15]
FOREIGN = ["TR", "AE", "DE", "GB", "US", "CN", "RU", "BR"]
HOUR_P_DAY = np.array([1, 1, 1, 1, 1, 2, 3, 5, 6, 6, 6, 7, 7, 7, 6, 6, 6, 7, 7, 7, 6, 4, 3, 2], float)
HOUR_P_NIGHT = np.array([6, 7, 7, 6, 5, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 5], float)


def _txn_frame(rng, card_idx, ts, amount, mcc, channel, foreign, device_new):
    n = len(card_idx)
    country = np.where(foreign, rng.choice(FOREIGN, n), "IR")
    distance = np.where(foreign, rng.uniform(500, 9000, n), rng.exponential(8, n))
    return pd.DataFrame(
        {
            "card_idx": card_idx, "timestamp": ts, "amount": amount.round(2), "merchant_category": mcc,
            "channel": channel, "merchant_country": country, "distance_from_home_km": distance.round(1),
            "device_new": device_new.astype(int),
        }
    )


def generate_fraud(rng, n_cards, days, txns_per_day, fraud_rate):
    start = pd.Timestamp("2025-01-01")
    card_avg = rng.lognormal(3.4, 0.5, n_cards)
    card_rate = rng.lognormal(np.log(txns_per_day), 0.6, n_cards)

    # legitimate traffic
    counts = rng.poisson(card_rate * days)
    card_idx = np.repeat(np.arange(n_cards), counts)
    n = len(card_idx)
    hour = rng.choice(24, n, p=HOUR_P_DAY / HOUR_P_DAY.sum())
    ts = start + pd.to_timedelta(rng.integers(0, days, n) * 86400 + hour * 3600 + rng.integers(0, 3600, n), unit="s")
    mcc = rng.choice(MCC, n, p=MCC_P)
    channel = np.where(mcc == "atm_cash", "atm", np.where(rng.random(n) < 0.3, "ecom", "pos"))
    legit = _txn_frame(
        rng, card_idx, ts, rng.lognormal(np.log(card_avg[card_idx]), 0.8), mcc, channel,
        rng.random(n) < 0.04, (channel == "ecom") & (rng.random(n) < 0.05),
    )
    legit["is_fraud"] = 0

    # fraud bursts on compromised cards
    n_bursts = max(1, int(n * fraud_rate / 3))
    frames = []
    for card in rng.integers(0, n_cards, n_bursts):
        size = 1 + rng.poisson(2)
        h = rng.choice(24, p=HOUR_P_NIGHT / HOUR_P_NIGHT.sum())
        t0 = start + pd.Timedelta(seconds=int(rng.integers(0, days) * 86400 + h * 3600 + rng.integers(0, 3600)))
        t = t0 + pd.to_timedelta(np.cumsum(rng.exponential(480, size)), unit="s")
        amount = rng.lognormal(np.log(card_avg[card] * 3), 0.9, size)
        probe = rng.random(size) < 0.2  # small card-testing charges
        amount = np.where(probe, rng.uniform(1, 5, size), amount)
        m = rng.choice(FRAUD_MCC, size, p=FRAUD_MCC_P)
        ch = np.where(m == "atm_cash", "atm", np.where(rng.random(size) < 0.7, "ecom", "pos"))
        f = _txn_frame(
            rng, np.full(size, card), t, amount, m, ch, rng.random(size) < 0.6,
            (ch == "ecom") & (rng.random(size) < 0.7),
        )
        f["is_fraud"] = 1
        frames.append(f)

    df = pd.concat([legit, *frames], ignore_index=True).sort_values(["card_idx", "timestamp"], kind="stable").reset_index(drop=True)
    df["card_id"] = "CARD" + df["card_idx"].astype(str).str.zfill(6)
    df["card_present"] = (df["channel"] != "ecom").astype(int)
    df["hour"] = df["timestamp"].dt.hour
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    df["is_foreign"] = (df["merchant_country"] != "IR").astype(int)

    # features computable at scoring time: only strictly earlier transactions of the same card
    g = df.groupby("card_idx")
    df["secs_since_last_txn"] = g["timestamp"].diff().dt.total_seconds()
    indexed = df.set_index("timestamp")
    roll = lambda col, w, fn: getattr(indexed.groupby("card_idx")[col].rolling(w, closed="left"), fn)().to_numpy()
    df["txn_count_1h"] = np.nan_to_num(roll("amount", "1h", "count"))
    df["txn_count_24h"] = np.nan_to_num(roll("amount", "24h", "count"))
    df["amount_sum_24h"] = np.nan_to_num(roll("amount", "24h", "sum")).round(2)
    prev_n = g.cumcount()
    prev_mean = (g["amount"].cumsum() - df["amount"]) / prev_n.replace(0, np.nan)
    df["amount_vs_card_avg"] = (df["amount"] / prev_mean).fillna(1.0).round(3)  # vs history so far

    df["transaction_id"] = [f"TXN{i:08d}" for i in range(len(df))]
    cols = [
        "transaction_id", "card_id", "timestamp", "amount", "merchant_category", "channel", "card_present",
        "merchant_country", "is_foreign", "distance_from_home_km", "hour", "is_night", "device_new",
        "secs_since_last_txn", "txn_count_1h", "txn_count_24h", "amount_sum_24h", "amount_vs_card_avg", "is_fraud",
    ]
    return df[cols].sort_values("timestamp", kind="stable").reset_index(drop=True)


def fraud_contributions(df):
    return pd.DataFrame(
        {
            "AMOUNT_ABOVE_CARD_NORM": 0.9 * np.clip(np.log(df["amount_vs_card_avg"].clip(lower=0.05)), -1, 3),
            "TRANSACTION_BURST": 0.7 * df["txn_count_1h"].clip(upper=5),
            "FOREIGN_MERCHANT": 1.4 * df["is_foreign"],
            "CARD_NOT_PRESENT": 0.5 * (1 - df["card_present"]),
            "NIGHT_TIME": 0.6 * df["is_night"],
            "NEW_DEVICE": 1.2 * df["device_new"],
            "HIGH_24H_SPEND": 0.4 * np.log1p(df["amount_sum_24h"] / 100),
        }
    )


# ---------------------------------------------------------------- decisions + audit


def credit_decisions(rng, df, contribs, logit, live_share, thresholds=(0.08, 0.20)):
    n_live = int(len(df) * live_share)
    live = df.index[-n_live:]
    score = sigmoid(logit[live] + rng.normal(0, 0.5, n_live))  # model = signal + estimation noise
    approve_below, decline_above = thresholds
    rec = np.where(score < approve_below, "approve", np.where(score > decline_above, "decline", "refer"))

    # analysts see every referral and a 10% sample of the automatic outcomes
    reviewed = (rec == "refer") | (rng.random(n_live) < 0.10)
    judgment = sigmoid(logit[live] + rng.normal(0, 0.4, n_live))  # analyst has extra context
    human = np.where(judgment < 0.14, "approve", "decline")
    override = reviewed & (rec != "refer") & (human != rec)
    final = np.where(reviewed, np.where(rec == "refer", human, np.where(override, human, rec)), rec)
    reasons = np.array(["", "income_verified_in_person", "additional_collateral", "documented_one_off_delinquency", "adverse_media_or_fraud_suspicion"])
    override_reason = np.where(override, rng.choice(reasons[1:], n_live), "")
    scored_at = df.loc[live, "application_ts"] + pd.to_timedelta(rng.integers(1, 4, n_live), unit="s")
    reviewed_at = scored_at + pd.to_timedelta(rng.exponential(4 * 3600, n_live), unit="s")

    out = pd.DataFrame(
        {
            "application_id": df.loc[live, "application_id"].to_numpy(),
            "model_version": "pd-v1.0",
            "scored_at": scored_at.to_numpy(),
            "pd_score": score.round(4),
            "model_recommendation": rec,
            "top_reasons": top_reasons(contribs.loc[live]),
            "reviewer_id": np.where(reviewed, rng.choice([f"analyst_{i:02d}" for i in range(1, 9)], n_live), ""),
            "reviewed_at": np.where(reviewed, reviewed_at, pd.NaT),
            "analyst_decision": np.where(reviewed, human, ""),
            "is_override": override.astype(int),
            "override_reason": override_reason,
            "final_decision": final,
        }
    )
    out["reviewed_at"] = pd.to_datetime(out["reviewed_at"])
    return out


def fraud_alerts(rng, tx, review_threshold, block_threshold):
    contribs = fraud_contributions(tx)
    z = contribs.sum(axis=1).to_numpy() - 4.0 + rng.normal(0, 0.8, len(tx))
    score = sigmoid(z)
    flagged = score >= review_threshold
    a = tx.loc[flagged, ["transaction_id", "card_id", "timestamp", "amount", "is_fraud"]].copy()
    s = score[flagged]
    n = len(a)
    a["model_version"] = "fraud-v1.0"
    a["fraud_score"] = s.round(4)
    a["scoring_latency_ms"] = np.clip(rng.lognormal(np.log(15), 0.35, n), 3, 49).round(1)
    a["action"] = np.where(s >= block_threshold, "block", "hold_for_review")
    a["top_reasons"] = top_reasons(contribs[flagged])
    # every hold is reviewed; blocks are spot-checked
    reviewed = (a["action"] == "hold_for_review").to_numpy() | (rng.random(n) < 0.25)
    correct = rng.random(n) < 0.93
    verdict = np.where(correct, a["is_fraud"], 1 - a["is_fraud"])
    a["reviewer_id"] = np.where(reviewed, rng.choice([f"analyst_{i:02d}" for i in range(9, 15)], n), "")
    a["reviewed_at"] = pd.to_datetime(
        np.where(reviewed, a["timestamp"] + pd.to_timedelta(rng.exponential(25 * 60, n), unit="s"), pd.NaT)
    )
    a["analyst_verdict"] = np.where(reviewed, np.where(verdict == 1, "confirmed_fraud", "false_positive"), "")
    return a.drop(columns="is_fraud").reset_index(drop=True), flagged


def build_audit_log(cd, fa):
    rows = []
    for r in cd.itertuples():
        base = {"entity_type": "loan_application", "entity_id": r.application_id}
        rows.append({**base, "ts": r.scored_at, "actor": r.model_version, "action": "MODEL_SCORED",
                     "details": {"pd_score": r.pd_score, "recommendation": r.model_recommendation}})
        rows.append({**base, "ts": r.scored_at, "actor": r.model_version, "action": "EXPLANATION_GENERATED",
                     "details": {"top_reasons": json.loads(r.top_reasons)}})
        if r.reviewer_id:
            rows.append({**base, "ts": r.reviewed_at, "actor": r.reviewer_id,
                         "action": "OVERRIDE" if r.is_override else "ANALYST_DECISION",
                         "details": {"decision": r.analyst_decision, "final": r.final_decision, "override_reason": r.override_reason}})
    for r in fa.itertuples():
        base = {"entity_type": "card_transaction", "entity_id": r.transaction_id}
        rows.append({**base, "ts": r.timestamp, "actor": r.model_version, "action": "MODEL_SCORED",
                     "details": {"fraud_score": r.fraud_score, "action": r.action, "latency_ms": r.scoring_latency_ms}})
        rows.append({**base, "ts": r.timestamp, "actor": r.model_version, "action": "EXPLANATION_GENERATED",
                     "details": {"top_reasons": json.loads(r.top_reasons)}})
        if r.reviewer_id:
            rows.append({**base, "ts": r.reviewed_at, "actor": r.reviewer_id, "action": "ANALYST_DECISION",
                         "details": {"verdict": r.analyst_verdict}})

    log = pd.DataFrame(rows).sort_values("ts", kind="stable").reset_index(drop=True)
    log.insert(0, "event_id", [f"EVT{i:08d}" for i in range(len(log))])
    log["details"] = log["details"].map(lambda d: json.dumps(d, sort_keys=True))

    # tamper-evident chain: each hash covers the previous hash and the full event content
    prev, hashes = "0" * 64, []
    for r in log.itertuples(index=False):
        payload = "|".join(str(x) for x in (r.event_id, r.ts, r.entity_type, r.entity_id, r.actor, r.action, r.details))
        prev = hashlib.sha256((prev + payload).encode()).hexdigest()
        hashes.append(prev)
    log["event_hash"] = hashes
    return log


# ---------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--n-applications", type=int, default=50_000)
    p.add_argument("--default-rate", type=float, default=0.09)
    p.add_argument("--live-share", type=float, default=0.2, help="latest share of applications that get decisions")
    p.add_argument("--n-cards", type=int, default=3_000)
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--txns-per-day", type=float, default=1.2)
    p.add_argument("--fraud-rate", type=float, default=0.004)
    p.add_argument("--review-threshold", type=float, default=0.30)
    p.add_argument("--block-threshold", type=float, default=0.70)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    credit, contribs, logit = generate_credit(rng, args.n_applications, args.default_rate)
    tx = generate_fraud(rng, args.n_cards, args.days, args.txns_per_day, args.fraud_rate)
    decisions = credit_decisions(rng, credit, contribs, logit, args.live_share)
    alerts, flagged = fraud_alerts(rng, tx, args.review_threshold, args.block_threshold)
    audit = build_audit_log(decisions, alerts)

    for name, frame in {
        "credit_applications": credit, "card_transactions": tx, "credit_decisions": decisions,
        "fraud_alerts": alerts, "audit_log": audit,
    }.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
        print(f"{name:<22}{len(frame):>9} rows")

    print(f"\ndefault rate : {credit['default'].mean():.3%}")
    print(f"fraud rate   : {tx['is_fraud'].mean():.3%}")
    print(f"fraud alerts : {flagged.sum()} flagged, recall {tx.loc[flagged, 'is_fraud'].sum() / tx['is_fraud'].sum():.1%}, "
          f"precision {tx.loc[flagged, 'is_fraud'].mean():.1%}")
    print(f"overrides    : {decisions['is_override'].sum()} of {len(decisions)} credit decisions")
    print(f"saved to     : {args.out_dir}/")


if __name__ == "__main__":
    main()
