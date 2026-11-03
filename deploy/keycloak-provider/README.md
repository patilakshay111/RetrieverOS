# RetrieverOS Keycloak provider

Custom Keycloak authentication SPIs that gate Google sign-in on a **verified** Google email address and auto-link brokered accounts to existing users.

## Why

The RetrieverOS realm configures the Google identity provider with `trustEmail: true` so that a Google sign-in can link to (or create) a workspace account using the Google-asserted email. Google only shares `email_verified: true` for addresses Google has actually verified, so we use that claim as the trust anchor. Without this gate, Keycloak would auto-link on any asserted email — an account-takeover vector for providers that assert unverified emails, and a source of junk accounts.

This provider was reconstructed from the previously built `retrieveros-google-broker.jar` (v1.0.0) and is maintained as source in this repository.

## What it provides

| Provider ID | Display type | Behavior |
| --- | --- | --- |
| `retrieveros-google-email-gate` | Verified Google email gate | Rejects the broker login (HTTP 403) unless the Google identity's `google_email_verified` attribute is exactly `["true"]` and the email is non-blank. |
| `retrieveros-google-email-auto-link` | Verified Google email auto-link | Same gate, then links the brokered identity to the existing enabled user referenced by the `EXISTING_USER_INFO` note only when its email matches the verified Google email (case/whitespace-insensitive). |

Both are Google-only: any other provider alias is rejected. If you add another IdP, create a separate broker flow for it.

## Broker flow

The realm ships a `retrieveros first broker login` flow (imported from `deploy/keycloak/retrieveros-realm.json` and selected as the realm's `firstBrokerLoginFlow`):

1. **Verified Google email gate** (REQUIRED) — hard stop for unverified identities.
2. **Review Profile** (REQUIRED, `review profile config`) — stock profile confirmation; silent when Google's profile is complete.
3. **User creation or linking** (REQUIRED subflow)
   - **Create User If Unique** (ALTERNATIVE, `create unique user config`) — creates a fresh user when no account shares the verified email.
   - **RetrieverOS handle existing account** (ALTERNATIVE subflow)
     - **Verified Google email auto-link** (REQUIRED) — links the Google identity to the existing same-email account. No password/email challenge is needed because Google already verified the mailbox.

Note: the classic "verify by email / re-authenticate" fallback was intentionally dropped for this flow. A Google identity without a verified, matching email is rejected outright.

## Build

The provider is built by `deploy/keycloak/Dockerfile` as a multi-stage Docker build (Maven 21 → `quay.io/keycloak/keycloak:26.3.2`), so no local Maven is required:

```powershell
docker compose build keycloak
```

To iterate locally with Maven:

```powershell
mvn -f deploy/keycloak-provider/pom.xml verify
```
