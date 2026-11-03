/*
 * RetrieverOS Keycloak provider: factory for the verified-Google-email gate.
 *
 * Reconstructed from the released retrieveros-google-broker.jar (v1.0.0).
 * License: Apache-2.0 (same as upstream Keycloak).
 */
package com.retrieveros.keycloak.auth;

import java.util.List;
import org.keycloak.Config;
import org.keycloak.authentication.Authenticator;
import org.keycloak.authentication.AuthenticatorFactory;
import org.keycloak.models.AuthenticationExecutionModel;
import org.keycloak.models.KeycloakSession;
import org.keycloak.models.KeycloakSessionFactory;
import org.keycloak.provider.ProviderConfigProperty;

public final class GoogleVerifiedEmailGateAuthenticatorFactory implements AuthenticatorFactory {
    public static final String PROVIDER_ID = "retrieveros-google-email-gate";
    private static final AuthenticationExecutionModel.Requirement[] REQUIREMENT_CHOICES = {
        AuthenticationExecutionModel.Requirement.REQUIRED,
        AuthenticationExecutionModel.Requirement.DISABLED
    };

    @Override
    public Authenticator create(KeycloakSession session) {
        return new GoogleVerifiedEmailGateAuthenticator();
    }

    @Override
    public String getId() {
        return PROVIDER_ID;
    }

    @Override
    public String getDisplayType() {
        return "Verified Google email gate";
    }

    @Override
    public String getReferenceCategory() {
        return null;
    }

    @Override
    public boolean isConfigurable() {
        return false;
    }

    @Override
    public AuthenticationExecutionModel.Requirement[] getRequirementChoices() {
        return REQUIREMENT_CHOICES.clone();
    }

    @Override
    public boolean isUserSetupAllowed() {
        return false;
    }

    @Override
    public String getHelpText() {
        return "Requires a verified Google email from Keycloak's broker context.";
    }

    @Override
    public List<ProviderConfigProperty> getConfigProperties() {
        return List.of();
    }

    @Override
    public void init(Config.Scope config) {
        // no initialization required
    }

    @Override
    public void postInit(KeycloakSessionFactory factory) {
        // no post-initialization required
    }

    @Override
    public void close() {
        // nothing to release
    }
}
