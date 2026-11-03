/*
 * RetrieverOS Keycloak provider: auto-links a brokered Google identity to an
 * existing enabled Keycloak user only when the Google email is verified and
 * matches the existing user's email.
 *
 * Reconstructed from the released retrieveros-google-broker.jar (v1.0.0).
 * License: Apache-2.0 (same as upstream Keycloak).
 */
package com.retrieveros.keycloak.auth;

import org.keycloak.authentication.AuthenticationFlowContext;
import org.keycloak.authentication.authenticators.broker.AbstractIdpAuthenticator;
import org.keycloak.authentication.authenticators.broker.util.ExistingUserInfo;
import org.keycloak.authentication.authenticators.broker.util.SerializedBrokeredIdentityContext;
import org.keycloak.broker.provider.BrokeredIdentityContext;
import org.keycloak.models.KeycloakSession;
import org.keycloak.models.RealmModel;
import org.keycloak.models.UserModel;

public final class GoogleVerifiedEmailAutoLinkAuthenticator extends AbstractIdpAuthenticator {
    private static final String EXISTING_USER_INFO_NOTE = "EXISTING_USER_INFO";

    @Override
    protected void authenticateImpl(
            AuthenticationFlowContext context,
            SerializedBrokeredIdentityContext serializedContext,
            BrokeredIdentityContext brokerContext) {
        if (!GoogleVerifiedEmailGateAuthenticator.isVerifiedGoogleIdentity(serializedContext, brokerContext)) {
            GoogleVerifiedEmailGateAuthenticator.reject(context);
            return;
        }
        UserModel existingUser = resolveExistingUser(context);
        if (existingUser == null
                || brokerContext == null
                || !GoogleVerifiedEmailPolicy.emailsMatch(existingUser.getEmail(), brokerContext.getEmail())) {
            GoogleVerifiedEmailGateAuthenticator.reject(context);
            return;
        }
        context.setUser(existingUser);
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

    private UserModel resolveExistingUser(AuthenticationFlowContext context) {
        try {
            String serialized = context.getAuthenticationSession().getAuthNote(EXISTING_USER_INFO_NOTE);
            if (serialized == null) {
                return null;
            }
            ExistingUserInfo existingUserInfo = ExistingUserInfo.deserialize(serialized);
            if (existingUserInfo == null
                    || existingUserInfo.getExistingUserId() == null
                    || existingUserInfo.getExistingUserId().isBlank()) {
                return null;
            }
            UserModel user = context.getSession()
                    .users()
                    .getUserById(context.getRealm(), existingUserInfo.getExistingUserId());
            return user != null && user.isEnabled() ? user : null;
        } catch (RuntimeException exception) {
            return null;
        }
    }
}
