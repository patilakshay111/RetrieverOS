package com.retrieveros.keycloak.auth;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class GoogleVerifiedEmailPolicyTest {

    @Test
    void acceptsVerifiedGoogleIdentityWithSingleTrueAttribute() {
        assertTrue(GoogleVerifiedEmailPolicy.isVerifiedGoogleIdentity("google", List.of("true"), "dev@example.com"));
    }

    @Test
    void rejectsNonGoogleProviderAlias() {
        assertFalse(GoogleVerifiedEmailPolicy.isVerifiedGoogleIdentity("github", List.of("true"), "dev@example.com"));
    }

    @Test
    void rejectsMissingOrMultipleVerifiedAttributeValues() {
        assertFalse(GoogleVerifiedEmailPolicy.isVerifiedGoogleIdentity("google", null, "dev@example.com"));
        assertFalse(GoogleVerifiedEmailPolicy.isVerifiedGoogleIdentity("google", List.of(), "dev@example.com"));
        assertFalse(GoogleVerifiedEmailPolicy.isVerifiedGoogleIdentity("google", List.of("true", "false"), "dev@example.com"));
        assertFalse(GoogleVerifiedEmailPolicy.isVerifiedGoogleIdentity("google", List.of("false"), "dev@example.com"));
        assertFalse(GoogleVerifiedEmailPolicy.isVerifiedGoogleIdentity("google", List.of("TRUE"), "dev@example.com"));
    }

    @Test
    void rejectsBlankEmailsAndProviderAliases() {
        assertFalse(GoogleVerifiedEmailPolicy.isVerifiedGoogleIdentity(null, List.of("true"), "dev@example.com"));
        assertFalse(GoogleVerifiedEmailPolicy.isVerifiedGoogleIdentity("google", List.of("true"), null));
        assertFalse(GoogleVerifiedEmailPolicy.isVerifiedGoogleIdentity("google", List.of("true"), ""));
        assertFalse(GoogleVerifiedEmailPolicy.isVerifiedGoogleIdentity("google", List.of("true"), "   "));
    }

    @Test
    void emailsMatchIsCaseAndWhitespaceInsensitive() {
        assertTrue(GoogleVerifiedEmailPolicy.emailsMatch("Dev@Example.com", " dev@example.com "));
        assertTrue(GoogleVerifiedEmailPolicy.emailsMatch("a@b.co", "A@b.co"));
    }

    @Test
    void emailsMismatchOrBlankNeverMatch() {
        assertFalse(GoogleVerifiedEmailPolicy.emailsMatch("a@b.co", "c@b.co"));
        assertFalse(GoogleVerifiedEmailPolicy.emailsMatch(null, "a@b.co"));
        assertFalse(GoogleVerifiedEmailPolicy.emailsMatch("a@b.co", null));
        assertFalse(GoogleVerifiedEmailPolicy.emailsMatch("", "a@b.co"));
        assertFalse(GoogleVerifiedEmailPolicy.emailsMatch("a@b.co", ""));
    }
}
