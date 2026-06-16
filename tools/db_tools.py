"""
tools/db_tools.py - SQLite query functions used by the database lookup node.

All functions return plain dicts/lists suitable for LLM tool results.
On any DB error, they return {"error": "..."} rather than raising, so a
single bad lookup never crashes the graph - the LLM can react to the
error message instead.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from core.logging_setup import get_logger
from core.settings import paths

logger = get_logger(__name__)


@contextmanager
def _connection():
    conn = sqlite3.connect(paths.db_path)
    try:
        yield conn
    finally:
        conn.close()


def _rows_to_dicts(cursor: sqlite3.Cursor, rows: list) -> List[Dict[str, Any]]:
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, row)) for row in rows]


def _row_to_dict(cursor: sqlite3.Cursor, row) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    columns = [d[0] for d in cursor.description]
    return dict(zip(columns, row))


def get_policy_details(policy_number: str) -> Dict[str, Any]:
    """Fetch policy details joined with customer info."""
    try:
        with _connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT p.*, c.first_name, c.last_name
                FROM policies p
                JOIN customers c ON p.customer_id = c.customer_id
                WHERE p.policy_number = ?
                """,
                (policy_number,),
            )
            row = cur.fetchone()
            result = _row_to_dict(cur, row)
        if result:
            logger.info("Policy found: %s", policy_number)
            return result
        logger.warning("Policy not found: %s", policy_number)
        return {"error": f"No policy found with number '{policy_number}'."}
    except sqlite3.Error as exc:
        logger.exception("Database error in get_policy_details")
        return {"error": f"Database error while looking up policy: {exc}"}


def get_auto_policy_details(policy_number: str) -> Dict[str, Any]:
    """Fetch auto-specific policy details including vehicle and deductibles."""
    try:
        with _connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT apd.*, p.policy_type, p.premium_amount
                FROM auto_policy_details apd
                JOIN policies p ON apd.policy_number = p.policy_number
                WHERE apd.policy_number = ?
                """,
                (policy_number,),
            )
            row = cur.fetchone()
            result = _row_to_dict(cur, row)
        if result:
            logger.info("Auto policy details found: %s", policy_number)
            return result
        logger.warning("Auto policy details not found: %s", policy_number)
        return {"error": f"No auto policy details found for '{policy_number}'."}
    except sqlite3.Error as exc:
        logger.exception("Database error in get_auto_policy_details")
        return {"error": f"Database error while looking up auto policy: {exc}"}


def get_claim_status(
    claim_id: Optional[str] = None,
    policy_number: Optional[str] = None,
) -> Any:
    """Get claim status by claim_id, or the latest 3 claims for a policy_number."""
    if not claim_id and not policy_number:
        return {"error": "Provide either a claim ID or a policy number."}

    try:
        with _connection() as conn:
            cur = conn.cursor()
            if claim_id:
                cur.execute(
                    """
                    SELECT c.*, p.policy_type
                    FROM claims c
                    JOIN policies p ON c.policy_number = p.policy_number
                    WHERE c.claim_id = ?
                    """,
                    (claim_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT c.*, p.policy_type
                    FROM claims c
                    JOIN policies p ON c.policy_number = p.policy_number
                    WHERE c.policy_number = ?
                    ORDER BY c.claim_date DESC
                    LIMIT 3
                    """,
                    (policy_number,),
                )
            rows = cur.fetchall()
            results = _rows_to_dicts(cur, rows)

        if results:
            logger.info("Found %d claim(s)", len(results))
            return results
        logger.warning("No claims found (claim_id=%s, policy=%s)", claim_id, policy_number)
        return {"error": "No claims found matching that information."}
    except sqlite3.Error as exc:
        logger.exception("Database error in get_claim_status")
        return {"error": f"Database error while looking up claim: {exc}"}


def get_billing_info(
    policy_number: Optional[str] = None,
    customer_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Get the most recent pending billing record for a policy or customer."""
    if not policy_number and not customer_id:
        return {"error": "Provide either a policy number or a customer ID."}

    try:
        with _connection() as conn:
            cur = conn.cursor()
            if policy_number:
                cur.execute(
                    """
                    SELECT b.*, p.premium_amount, p.billing_frequency
                    FROM billing b
                    JOIN policies p ON b.policy_number = p.policy_number
                    WHERE b.policy_number = ? AND b.status = 'pending'
                    ORDER BY b.due_date DESC
                    LIMIT 1
                    """,
                    (policy_number,),
                )
            else:
                cur.execute(
                    """
                    SELECT b.*, p.premium_amount, p.billing_frequency
                    FROM billing b
                    JOIN policies p ON b.policy_number = p.policy_number
                    WHERE p.customer_id = ? AND b.status = 'pending'
                    ORDER BY b.due_date DESC
                    LIMIT 1
                    """,
                    (customer_id,),
                )
            row = cur.fetchone()
            result = _row_to_dict(cur, row)

        if result:
            logger.info("Billing info found")
            return result
        logger.warning("No pending billing info found")
        return {"error": "No pending billing information found."}
    except sqlite3.Error as exc:
        logger.exception("Database error in get_billing_info")
        return {"error": f"Database error while looking up billing info: {exc}"}


def get_payment_history(policy_number: str) -> List[Dict[str, Any]]:
    """Get the last 10 payments for a policy."""
    try:
        with _connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT p.payment_date, p.amount, p.status, p.payment_method
                FROM payments p
                JOIN billing b ON p.bill_id = b.bill_id
                WHERE b.policy_number = ?
                ORDER BY p.payment_date DESC
                LIMIT 10
                """,
                (policy_number,),
            )
            rows = cur.fetchall()
            results = _rows_to_dicts(cur, rows)

        if results:
            logger.info("Found %d payment record(s)", len(results))
        else:
            logger.warning("No payment history found for policy: %s", policy_number)
        return results
    except sqlite3.Error as exc:
        logger.exception("Database error in get_payment_history")
        return [{"error": f"Database error while looking up payment history: {exc}"}]


def get_renewal_status(policy_number: str) -> Dict[str, Any]:
    """Get renewal-relevant info for a policy (status, start date, type)."""
    try:
        with _connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT policy_number, policy_type, start_date, status, premium_amount
                FROM policies
                WHERE policy_number = ?
                """,
                (policy_number,),
            )
            row = cur.fetchone()
            result = _row_to_dict(cur, row)

        if result:
            logger.info("Renewal info found: %s", policy_number)
            return result
        logger.warning("Policy not found for renewal lookup: %s", policy_number)
        return {"error": f"No policy found with number '{policy_number}'."}
    except sqlite3.Error as exc:
        logger.exception("Database error in get_renewal_status")
        return {"error": f"Database error while checking renewal status: {exc}"}
