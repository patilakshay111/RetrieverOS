export interface AuthConfig {
  url: string;
  realm: string;
  clientId: string;
}

export const authConfig: AuthConfig = {
  url: (import.meta.env.VITE_KEYCLOAK_URL ?? "http://localhost:8080").replace(/\/+$/, ""),
  realm: import.meta.env.VITE_KEYCLOAK_REALM ?? "retrieveros",
  clientId: import.meta.env.VITE_KEYCLOAK_CLIENT_ID ?? "retrieveros-web",
};
