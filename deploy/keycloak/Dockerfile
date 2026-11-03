# RetrieverOS Keycloak image: stock Keycloak 26.3.2 + the RetrieverOS
# Google-verified-email broker provider built from source (see
# deploy/keycloak-provider/). Build context is the repository root.

FROM maven:3.9-eclipse-temurin-21 AS provider-build
WORKDIR /build
COPY deploy/keycloak-provider/pom.xml .
RUN mvn -B -q dependency:go-offline
COPY deploy/keycloak-provider/src ./src
RUN mvn -B -q -DskipTests package

FROM quay.io/keycloak/keycloak:26.3.2
COPY --from=provider-build /build/target/retrieveros-keycloak-provider-1.0.0.jar /opt/keycloak/providers/retrieveros-google-broker.jar
RUN /opt/keycloak/bin/kc.sh build
