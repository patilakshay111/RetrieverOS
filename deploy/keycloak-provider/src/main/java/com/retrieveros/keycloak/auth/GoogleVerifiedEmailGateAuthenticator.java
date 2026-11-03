/*
 * RetrieverOS Keycloak provider: rejects Google broker logins unless the
 * Google identity carries email_verified=true.
 *
 * Reconstructed from the released retrieveros-google-broker.jar (v1.0.0).
 * License: Apache-2.0 (same as upstream Keycloak).
 */
package com.retrieveros.keycloak.auth;

import jakarta.ws.rs.core.Response;
import org.keycloak.authentication.AuthenticationFlowContext;
import org.keycloak.authentication.AuthenticationFlowError;
import org.keycloak.authentication.authenticators.broker.AbstractIdpAuthenticator;
import org.keycloak.authentication.authenticators.broker.util.SerializedBrokeredIdentityContext;
import org.keycloak.broker.provider.BrokeredIdentityContext;
import org.keycloak.models.KeycloakSession;
import org.keycloak.models.RealmModel;
import org.keycloak.models.UserModel;

public final class GoogleVerifiedEmailGateAuthenticator extends AbstractIdpAuthenticator {
    @Override
    protected void authenticateImpl(
            AuthenticationFlowContext context,
            SerializedBrokeredIdentityContext serializedContext,
            BrokeredIdentityContext brokerContext) {
        if (!isVerifiedGoogleIdentity(serializedContext, brokerContext)) {
            reject(context);
            return;
        }
        context.success();
    }

    @Override
    protected void actionImpl(
            AuthenticationFlowContext context,
            SerializedBrokeredIdentityContext serializedContext,
            BrokeredIdentityContext brokerContext) {
        authenticateImpl(context, serializedContext, brokerContext);
    }

    @Override
    public boolean requiresUser() {
        return false;
    }

    @Override
    public boolean configuredFor(KeycloakSession session, RealmModel realm, UserModel user) {
        return false;
    }

    static boolean isVerifiedGoogleIdentity(
            SerializedBrokeredIdentityContext serializedContext, BrokeredIdentityContext brokerContext) {
        String providerAlias = brokerContext == null || brokerContext.getIdpConfig() == null
                ? null
                : brokerContext.getIdpConfig().getAlias();
        String email = brokerContext == null ? null : brokerContext.getEmail();
        return serializedContext != null
                && GoogleVerifiedEmailPolicy.isVerifiedGoogleIdentity(
                        providerAlias, serializedContext.getAttribute("google_email_verified"), email);
    }

    static void reject(AuthenticationFlowContext context) {
        Response response = context.form()
                .setError("identityProviderUnexpectedErrorMessage", new Object[0])
                .createErrorPage(Response.Status.FORBIDDEN);
        context.failureChallenge(AuthenticationFlowError.IDENTITY_PROVIDER_ERROR, response);
    }
}
