"""Authentication helpers used as a RAG/index test fixture."""

import hashlib


def hash_password(password, salt):
    """Hash a password with a salt using SHA-256."""
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def verify_password(password, salt, expected_hash):
    """Return True if the password matches the expected hash."""
    return hash_password(password, salt) == expected_hash


class SessionManager:
    """In-memory session token registry."""

    def __init__(self):
        self.sessions = {}

    def create_session(self, user_id, token):
        self.sessions[token] = user_id
        return token

    def get_user(self, token):
        return self.sessions.get(token)
