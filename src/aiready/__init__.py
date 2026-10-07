"""AI-Ready: an AI foundation deployed beside existing applications.

Every service in this package follows the same rules:
  * the existing application is never modified - it is reached through its APIs, database replica and login;
  * every model call goes through the gateway;
  * every call carries the signed-in user's identity;
  * every decision is traced and written to the audit log.
"""

__version__ = "0.1.0"
