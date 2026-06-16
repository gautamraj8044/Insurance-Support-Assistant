"""
data/generator.py - Generates synthetic insurance data for all DB tables.
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


_FIRST_NAMES = [
    "John", "Jane", "Robert", "Maria", "David", "Lisa", "Michael", "Sarah", "James", "Emily",
    "William", "Emma", "Joseph", "Olivia", "Charles", "Ava", "Thomas", "Isabella", "Daniel", "Mia",
    "Matthew", "Sophia", "Anthony", "Charlotte", "Christopher", "Amelia", "Andrew", "Harper",
    "Joshua", "Evelyn", "Ryan", "Abigail", "Brandon", "Ella", "Justin", "Scarlett", "Tyler", "Grace",
    "Alexander", "Chloe", "Kevin", "Victoria", "Jason", "Lily", "Brian", "Hannah", "Eric", "Aria",
    "Kyle", "Zoey",
]

_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts",
]


def generate_sample_data(random_state: int = 42) -> dict[str, pd.DataFrame]:
    """Return a dict of DataFrames keyed by table name."""
    random.seed(random_state)
    np.random.seed(random_state)

    n_customers = 1000
    n_policies = 1500
    n_billing = 5000
    n_payments = 4000
    n_claims = 300

    customers = pd.DataFrame(
        {
            "customer_id": [f"CUST{str(i).zfill(5)}" for i in range(1, n_customers + 1)],
            "first_name": [random.choice(_FIRST_NAMES) for _ in range(n_customers)],
            "last_name": [random.choice(_LAST_NAMES) for _ in range(n_customers)],
            "email": [f"user{i}@example.com" for i in range(1, n_customers + 1)],
            "phone": [
                f"555-{random.randint(100, 999):03d}-{random.randint(1000, 9999):04d}"
                for _ in range(n_customers)
            ],
            "date_of_birth": [
                datetime(1980, 1, 1) + timedelta(days=random.randint(0, 10_000))
                for _ in range(n_customers)
            ],
            "state": [
                random.choice(["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA"])
                for _ in range(n_customers)
            ],
        }
    )

    policies = pd.DataFrame(
        {
            "policy_number": [f"POL{str(i).zfill(6)}" for i in range(1, n_policies + 1)],
            "customer_id": [
                f"CUST{random.randint(1, n_customers):05d}" for _ in range(n_policies)
            ],
            "policy_type": [
                random.choice(["auto", "home", "life"]) for _ in range(n_policies)
            ],
            "start_date": [
                datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))
                for _ in range(n_policies)
            ],
            "premium_amount": [round(random.uniform(50, 500), 2) for _ in range(n_policies)],
            "billing_frequency": [
                random.choice(["monthly", "quarterly", "annual"]) for _ in range(n_policies)
            ],
            "status": [
                random.choice(["active", "active", "active", "cancelled"])
                for _ in range(n_policies)
            ],
        }
    )

    auto_policies = policies[policies["policy_type"] == "auto"].copy()
    n_auto = len(auto_policies)
    auto_policy_details = pd.DataFrame(
        {
            "policy_number": auto_policies["policy_number"].values,
            "vehicle_vin": [
                f"VIN{random.randint(10_000_000_000_000_000, 99_999_999_999_999_999)}"
                for _ in range(n_auto)
            ],
            "vehicle_make": [
                random.choice(["Toyota", "Honda", "Ford", "Chevrolet", "Nissan"])
                for _ in range(n_auto)
            ],
            "vehicle_model": [
                random.choice(["Camry", "Civic", "F-150", "Malibu", "Altima"])
                for _ in range(n_auto)
            ],
            "vehicle_year": [random.randint(2015, 2023) for _ in range(n_auto)],
            "liability_limit": [
                random.choice([50_000, 100_000, 300_000]) for _ in range(n_auto)
            ],
            "collision_deductible": [
                random.choice([250, 500, 1000]) for _ in range(n_auto)
            ],
            "comprehensive_deductible": [
                random.choice([250, 500, 1000]) for _ in range(n_auto)
            ],
            "uninsured_motorist": [random.choice([0, 1]) for _ in range(n_auto)],
            "rental_car_coverage": [random.choice([0, 1]) for _ in range(n_auto)],
        }
    )

    billing = pd.DataFrame(
        {
            "bill_id": [f"BILL{str(i).zfill(6)}" for i in range(1, n_billing + 1)],
            "policy_number": [
                random.choice(policies["policy_number"]) for _ in range(n_billing)
            ],
            "billing_date": [
                datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
                for _ in range(n_billing)
            ],
            "due_date": [
                datetime(2024, 1, 15) + timedelta(days=random.randint(0, 90))
                for _ in range(n_billing)
            ],
            "amount_due": [round(random.uniform(100, 1000), 2) for _ in range(n_billing)],
            "status": [
                random.choice(["paid", "pending", "overdue"]) for _ in range(n_billing)
            ],
        }
    )

    payments = pd.DataFrame(
        {
            "payment_id": [f"PAY{str(i).zfill(6)}" for i in range(1, n_payments + 1)],
            "bill_id": [random.choice(billing["bill_id"]) for _ in range(n_payments)],
            "payment_date": [
                datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
                for _ in range(n_payments)
            ],
            "amount": [round(random.uniform(50, 500), 2) for _ in range(n_payments)],
            "payment_method": [
                random.choice(["credit_card", "debit_card", "bank_transfer"])
                for _ in range(n_payments)
            ],
            "transaction_id": [
                f"TXN{random.randint(100_000, 999_999)}" for _ in range(n_payments)
            ],
            "status": [
                random.choice(["completed", "pending", "failed"]) for _ in range(n_payments)
            ],
        }
    )

    claims = pd.DataFrame(
        {
            "claim_id": [f"CLM{str(i).zfill(6)}" for i in range(1, n_claims + 1)],
            "policy_number": [
                random.choice(policies["policy_number"]) for _ in range(n_claims)
            ],
            "claim_date": [
                datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
                for _ in range(n_claims)
            ],
            "incident_type": [
                random.choice(
                    ["collision", "theft", "property_damage", "medical", "liability"]
                )
                for _ in range(n_claims)
            ],
            "estimated_loss": [
                round(random.uniform(500, 20_000), 2) for _ in range(n_claims)
            ],
            "status": [
                random.choice(
                    ["submitted", "under_review", "approved", "paid", "denied"]
                )
                for _ in range(n_claims)
            ],
        }
    )

    return {
        "customers": customers,
        "policies": policies,
        "auto_policy_details": auto_policy_details,
        "billing": billing,
        "payments": payments,
        "claims": claims,
    }
