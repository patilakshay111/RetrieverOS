/*
 * RetrieverOS Keycloak provider: shared policy for Google-verified-email broker flows.
 *
 * Reconstructed from the released retrieveros-google-broker.jar (v1.0.0).
 * License: Apache-2.0 (same as upstream Keycloak).
 */
package com.retrieveros.keycloak.auth;

import java.util.List;
import java.util.Locale;

public final class GoogleVerifiedEmailPolicy {
    private static final String GOOGLE_PROVIDER_ALIAS = "google";
    private static final String VERIFIED_VALUE = "true";

    private GoogleVerifiedEmailPolicy() {
    }

    public static boolean isVerifiedGoogleIdentity(String providerAlias, List<String> verifiedValues, String email) {
        return GOOGLE_PROVIDER_ALIAS.equals(providerAlias)
                && verifiedValues != null
                && verifiedValues.size() == 1
                && VERIFIED_VALUE.equals(verifiedValues.get(0))
                && hasEmail(email);
    }

    public static boolean emailsMatch(String existingEmail, String googleEmail) {
        return hasEmail(existingEmail)
                && hasEmail(googleEmail)
                && normalize(existingEmail).equals(normalize(googleEmail));
    }

    private static boolean hasEmail(String email) {
        return email != null && !email.isBlank();
    }

    private static String normalize(String email) {
        return email.strip().toLowerCase(Locale.ROOT);
    }
}
