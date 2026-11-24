// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Stateless authentication shared by the HTTP gateways (search, Flight SQL).
//!
//! Two modes, both of which avoid maintaining a user database:
//!
//! * **API key** — a shared secret compared in constant time (`BDB_API_KEY`).
//! * **JWT** — a bearer token verified against a shared secret (HS256,
//!   `BDB_JWT_SECRET`) or an RSA public key (RS256, `BDB_JWT_PUBLIC_KEY`,
//!   PEM). This is the "delegate to your IdP" path: the IdP signs, we only
//!   verify the signature and read `exp`/`nbf`/`aud`/`iss`/`sub` claims.
//!
//! When neither is configured the guard is a no-op; set `BDB_AUTH_REQUIRED=1`
//! to fail closed instead.

use base64::Engine as _;
use serde_json::Value;
use subtle::ConstantTimeEq;

/// Extracted claims from an authenticated token.
#[derive(Clone, Debug, Default)]
pub struct AuthClaims {
    pub subject: String,
    pub roles: Vec<String>,
}

/// Authentication configuration, resolved once at startup.
#[derive(Clone, Default)]
pub struct AuthConfig {
    api_key: Option<String>,
    jwt_secret: Option<Vec<u8>>,
    jwt_public_key: Option<Vec<u8>>,
    jwt_audience: Option<String>,
    jwt_issuer: Option<String>,
    /// Fail closed when no credential is configured.
    required: bool,
    /// Exempt `/metrics` from auth (for an internal Prometheus scraper).
    metrics_public: bool,
}

impl AuthConfig {
    /// Build from environment variables.
    pub fn from_env() -> Self {
        let env_get = |name: &str| std::env::var(format!("BDB_{name}"));
        let truthy =
            |name: &str| matches!(env_get(name).as_deref(), Ok("1") | Ok("true") | Ok("yes"));
        Self {
            api_key: env_get("API_KEY").ok().filter(|s| !s.is_empty()),
            jwt_secret: env_get("JWT_SECRET")
                .ok()
                .filter(|s| !s.is_empty())
                .map(|s| s.into_bytes()),
            jwt_public_key: env_get("JWT_PUBLIC_KEY")
                .ok()
                .filter(|s| !s.is_empty())
                .map(|s| s.into_bytes()),
            jwt_audience: env_get("JWT_AUDIENCE").ok().filter(|s| !s.is_empty()),
            jwt_issuer: env_get("JWT_ISSUER").ok().filter(|s| !s.is_empty()),
            required: truthy("AUTH_REQUIRED"),
            metrics_public: truthy("METRICS_PUBLIC"),
        }
    }

    /// Whether any credential source is configured.
    pub fn enabled(&self) -> bool {
        self.api_key.is_some() || self.jwt_secret.is_some() || self.jwt_public_key.is_some()
    }

    /// Whether auth is required even when no credential is configured.
    pub fn required(&self) -> bool {
        self.required
    }

    /// Whether `/metrics` is exempt from auth.
    pub fn metrics_public(&self) -> bool {
        self.metrics_public
    }

    /// Override whether `/metrics` is exempt from auth (builder form).
    pub fn with_metrics_public(mut self, public: bool) -> Self {
        self.metrics_public = public;
        self
    }

    /// Builder: configure a shared API key programmatically (embedding/tests).
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Verify a bearer token (API key or JWT). Returns the claims on success.
    pub fn verify(&self, token: &str) -> Result<AuthClaims, String> {
        // API key first (constant-time).
        if let Some(expected) = &self.api_key {
            if expected.as_bytes().ct_eq(token.as_bytes()).into() {
                return Ok(AuthClaims {
                    subject: "api-key".to_string(),
                    roles: vec!["admin".to_string()],
                });
            }
        }
        // JWT.
        if self.jwt_secret.is_some() || self.jwt_public_key.is_some() {
            return self.verify_jwt(token);
        }
        Err("no credential configured".to_string())
    }

    fn verify_jwt(&self, token: &str) -> Result<AuthClaims, String> {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return Err("malformed JWT".to_string());
        }
        let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let header_bytes = engine
            .decode(parts[0])
            .map_err(|_| "bad JWT header encoding".to_string())?;
        let payload_bytes = engine
            .decode(parts[1])
            .map_err(|_| "bad JWT payload encoding".to_string())?;
        let signature = engine
            .decode(parts[2])
            .map_err(|_| "bad JWT signature encoding".to_string())?;

        let header: Value =
            serde_json::from_slice(&header_bytes).map_err(|_| "bad JWT header".to_string())?;
        let alg = header
            .get("alg")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "JWT missing alg".to_string())?;

        let signing_input = format!("{}.{}", parts[0], parts[1]);
        match alg {
            "HS256" => {
                let secret = self
                    .jwt_secret
                    .as_ref()
                    .ok_or_else(|| "HS256 token but no BDB_JWT_SECRET".to_string())?;
                verify_hs256(secret, signing_input.as_bytes(), &signature)?;
            }
            "RS256" => {
                let pem = self
                    .jwt_public_key
                    .as_ref()
                    .ok_or_else(|| "RS256 token but no BDB_JWT_PUBLIC_KEY".to_string())?;
                verify_rs256(pem, signing_input.as_bytes(), &signature)?;
            }
            other => return Err(format!("unsupported JWT alg '{other}'")),
        }

        let claims: Value =
            serde_json::from_slice(&payload_bytes).map_err(|_| "bad JWT payload".to_string())?;
        self.check_claims(&claims)?;

        let subject = claims
            .get("sub")
            .and_then(|v| v.as_str())
            .unwrap_or("jwt")
            .to_string();

        let roles = claims
            .get("roles")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        Ok(AuthClaims { subject, roles })
    }

    fn check_claims(&self, claims: &Value) -> Result<(), String> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let exp = claims
            .get("exp")
            .and_then(|v| v.as_f64())
            .map(|f| f as u64)
            .ok_or_else(|| "JWT missing exp claim".to_string())?;
        if now >= exp {
            return Err("JWT expired".to_string());
        }

        if let Some(nbf) = claims
            .get("nbf")
            .and_then(|v| v.as_f64().or_else(|| v.as_u64().map(|u| u as f64)))
            .map(|f| f as u64)
        {
            if now < nbf {
                return Err("JWT not yet valid".to_string());
            }
        }
        if let Some(expected) = &self.jwt_issuer {
            let iss = claims.get("iss").and_then(|v| v.as_str()).unwrap_or("");
            if iss != expected {
                return Err("JWT issuer mismatch".to_string());
            }
        }
        if let Some(expected) = &self.jwt_audience {
            let ok = match claims.get("aud") {
                Some(Value::String(s)) => s == expected,
                Some(Value::Array(a)) => a.iter().any(|v| v.as_str() == Some(expected.as_str())),
                _ => false,
            };
            if !ok {
                return Err("JWT audience mismatch".to_string());
            }
        }
        Ok(())
    }
}

/// HMAC-SHA256 verification (constant-time).
fn verify_hs256(secret: &[u8], signing_input: &[u8], signature: &[u8]) -> Result<(), String> {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    let mut mac =
        Hmac::<Sha256>::new_from_slice(secret).map_err(|_| "invalid HMAC key".to_string())?;
    mac.update(signing_input);
    mac.verify_slice(signature)
        .map_err(|_| "JWT signature mismatch".to_string())
}

/// RSA PKCS#1 v1.5 SHA-256 verification using a PEM public key.
fn verify_rs256(pem: &[u8], signing_input: &[u8], signature: &[u8]) -> Result<(), String> {
    use ring::signature::{UnparsedPublicKey, RSA_PKCS1_2048_8192_SHA256};
    let der = pem_to_pkcs1_der(pem)?;
    let key = UnparsedPublicKey::new(&RSA_PKCS1_2048_8192_SHA256, der);
    key.verify(signing_input, signature)
        .map_err(|_| "JWT signature mismatch".to_string())
}

/// Extract the DER bytes from a PEM `PUBLIC KEY` (SPKI) or `RSA PUBLIC KEY` (PKCS#1) block,
/// unwrapping the outer SPKI ASN.1 envelope if present to yield raw PKCS#1 `RSAPublicKey` DER.
fn pem_to_pkcs1_der(pem: &[u8]) -> Result<Vec<u8>, String> {
    let raw_der = pem_to_der(pem)?;
    Ok(extract_pkcs1_from_spki(&raw_der))
}

/// Unwrap X.509 SubjectPublicKeyInfo (SPKI) to extract the inner PKCS#1 `RSAPublicKey` DER.
/// If the input is already PKCS#1 or not a valid SPKI sequence, returns the input bytes unmodified.
fn extract_pkcs1_from_spki(der: &[u8]) -> Vec<u8> {
    if der.is_empty() || der[0] != 0x30 {
        return der.to_vec();
    }
    let mut idx = 1;
    // Skip sequence length
    if idx < der.len() && der[idx] & 0x80 != 0 {
        let num_len_bytes = (der[idx] & 0x7F) as usize;
        idx += 1 + num_len_bytes;
    } else {
        idx += 1;
    }
    if idx >= der.len() {
        return der.to_vec();
    }
    // If the first element is INTEGER (0x02), this is already an RSAPublicKey (modulus)
    if der[idx] == 0x02 {
        return der.to_vec();
    }
    // If first element is SEQUENCE (0x30), this is the AlgorithmIdentifier in SPKI
    if der[idx] == 0x30 {
        idx += 1;
        if idx < der.len() && der[idx] & 0x80 != 0 {
            let alg_len_bytes = (der[idx] & 0x7F) as usize;
            idx += 1;
            if idx + alg_len_bytes <= der.len() {
                let mut alg_len = 0usize;
                for &b in &der[idx..idx + alg_len_bytes] {
                    alg_len = (alg_len << 8) | (b as usize);
                }
                idx += alg_len_bytes + alg_len;
            } else {
                return der.to_vec();
            }
        } else if idx < der.len() {
            let alg_len = der[idx] as usize;
            idx += 1 + alg_len;
        } else {
            return der.to_vec();
        }

        // The next element in SPKI must be a BIT STRING (tag 0x03)
        if idx < der.len() && der[idx] == 0x03 {
            idx += 1;
            if idx < der.len() && der[idx] & 0x80 != 0 {
                let bs_len_bytes = (der[idx] & 0x7F) as usize;
                idx += 1 + bs_len_bytes;
            } else {
                idx += 1;
            }
            // First byte of BIT STRING value is unused bits count (must be 0x00 for byte-aligned key)
            if idx < der.len() && der[idx] == 0x00 {
                idx += 1;
                return der[idx..].to_vec();
            }
        }
    }
    der.to_vec()
}

/// Extract the DER bytes from a PEM `PUBLIC KEY` (SPKI) block.
fn pem_to_der(pem: &[u8]) -> Result<Vec<u8>, String> {
    let text = std::str::from_utf8(pem).map_err(|_| "public key is not UTF-8".to_string())?;
    let body: String = text
        .lines()
        .filter(|l| !l.starts_with("-----"))
        .collect::<Vec<_>>()
        .join("");
    base64::engine::general_purpose::STANDARD
        .decode(body.trim())
        .map_err(|_| "public key is not valid base64".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hs256_token(secret: &[u8], claims: &str) -> String {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let header = engine.encode(br#"{"alg":"HS256","typ":"JWT"}"#);
        let payload = engine.encode(claims.as_bytes());
        let signing_input = format!("{header}.{payload}");
        let mut mac = Hmac::<Sha256>::new_from_slice(secret).unwrap();
        mac.update(signing_input.as_bytes());
        let sig = engine.encode(mac.finalize().into_bytes());
        format!("{signing_input}.{sig}")
    }

    #[test]
    fn api_key_constant_time_match() {
        let cfg = AuthConfig {
            api_key: Some("s3cret".to_string()),
            ..Default::default()
        };
        assert!(cfg.verify("s3cret").is_ok());
        assert!(cfg.verify("wrong").is_err());
    }

    #[test]
    fn hs256_round_trip_and_expiry() {
        let secret = b"top-secret";
        let cfg = AuthConfig {
            jwt_secret: Some(secret.to_vec()),
            ..Default::default()
        };
        let future = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 3600;
        let good = hs256_token(secret, &format!(r#"{{"sub":"alice","exp":{future}}}"#));
        assert_eq!(cfg.verify(&good).unwrap().subject, "alice");

        let expired = hs256_token(secret, r#"{"sub":"bob","exp":1}"#);
        assert!(cfg.verify(&expired).is_err());

        let other = AuthConfig {
            jwt_secret: Some(b"different".to_vec()),
            ..Default::default()
        };
        assert!(other.verify(&good).is_err());
    }

    #[test]
    fn rbac_roles_extraction() {
        let secret = b"rbac-secret";
        let cfg = AuthConfig {
            jwt_secret: Some(secret.to_vec()),
            ..Default::default()
        };
        let future = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 3600;

        // Test JWT with roles
        let token_with_roles = hs256_token(
            secret,
            &format!(
                r#"{{"sub":"charlie","exp":{},"roles":["admin","data_engineer"]}}"#,
                future
            ),
        );
        let claims = cfg.verify(&token_with_roles).unwrap();
        assert_eq!(claims.subject, "charlie");
        assert_eq!(claims.roles.len(), 2);
        assert!(claims.roles.contains(&"admin".to_string()));
        assert!(claims.roles.contains(&"data_engineer".to_string()));

        // Test JWT without roles gracefully falls back to empty roles
        let token_without_roles =
            hs256_token(secret, &format!(r#"{{"sub":"david","exp":{}}}"#, future));
        let claims = cfg.verify(&token_without_roles).unwrap();
        assert_eq!(claims.subject, "david");
        assert!(claims.roles.is_empty());
    }

    #[test]
    fn audience_and_issuer_enforced() {
        let secret = b"k";
        let cfg = AuthConfig {
            jwt_secret: Some(secret.to_vec()),
            jwt_audience: Some("benodb".to_string()),
            jwt_issuer: Some("https://idp.example".to_string()),
            ..Default::default()
        };
        let future = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 60;
        let ok = hs256_token(
            secret,
            &format!(r#"{{"sub":"a","exp":{future},"aud":"benodb","iss":"https://idp.example"}}"#),
        );
        assert!(cfg.verify(&ok).is_ok());
        let bad_aud = hs256_token(
            secret,
            &format!(r#"{{"sub":"a","exp":{future},"aud":"other","iss":"https://idp.example"}}"#),
        );
        assert!(cfg.verify(&bad_aud).is_err());
    }

    #[test]
    fn rs256_verification_spki_and_pkcs1_keys() {
        let spki_pem = b"-----BEGIN PUBLIC KEY-----\n\
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA0wcMWXsyIXuWgVrm/iaH\n\
y+W1ivmdG/mnDKoUIHXywFk84EY4nj7BVorjnJDUz35+orQLJYzcSvOhbzLzkO/d\n\
yWkzQfJ/PzAgK4fJ18ws2SJMRSNWDrrjeSH48Icb5J6sHVklj7hqb0z2fzHg32mM\n\
F+GK1LNV0lHHb0VPar8mMNG6OFxPTY4sETQeDCt4dlE0xCoa6/Puh7TsPfTiuMJa\n\
oicXxJN1YrIU/LnPEyL/JD4yKmc4hXoFkZY/RRDhogsL0akhT8hR0eJaLb5mbUHl\n\
5TPDUNM1m4N9QNdqJxb8oiuqP60kLO2jTKnLDbCIHuK1633JiZHi18Ty8ziN3tIT\n\
LwIDAQAB\n\
-----END PUBLIC KEY-----\n";

        let pkcs1_pem = b"-----BEGIN RSA PUBLIC KEY-----\n\
MIIBCgKCAQEA0wcMWXsyIXuWgVrm/iaHy+W1ivmdG/mnDKoUIHXywFk84EY4nj7B\n\
VorjnJDUz35+orQLJYzcSvOhbzLzkO/dyWkzQfJ/PzAgK4fJ18ws2SJMRSNWDrrj\n\
eSH48Icb5J6sHVklj7hqb0z2fzHg32mMF+GK1LNV0lHHb0VPar8mMNG6OFxPTY4s\n\
ETQeDCt4dlE0xCoa6/Puh7TsPfTiuMJaoicXxJN1YrIU/LnPEyL/JD4yKmc4hXoF\n\
kZY/RRDhogsL0akhT8hR0eJaLb5mbUHl5TPDUNM1m4N9QNdqJxb8oiuqP60kLO2j\n\
TKnLDbCIHuK1633JiZHi18Ty8ziN3tITLwIDAQAB\n\
-----END RSA PUBLIC KEY-----\n";

        let valid_token = "eyJhbGciOiAiUlMyNTYiLCAidHlwIjogIkpXVCJ9.\
eyJzdWIiOiAiYWxpY2UtcnNhIiwgImV4cCI6IDE3OTU0OTExMjksICJyb2xlcyI6IFsiYWRtaW4iXX0.\
BHgPYSZixWc6ek-AOMf7Sidcao1TaL9WvDbrc-g7sj-EKQjAhxFrRWXcMKBpbcOk_Ow14lAmZ5lNbbqqYsHk2EpjKwNsz7VJblNKdvM08d_TGppfmcQNhG2YM4avXGOS-98tMBrhCNCufdjxc28F0-vu6fKEzEQtXmo3EcGKkXDX1ZIEQ4xF8XbcK9Qr2HW0X08kJ-alRDR5QfRS5uMk1KBgNR54AZluX9ZdDSnm9VNiymTzT4aax3tIDE_rA52jERg21WRD9rX2Bs0BbpS8nUjJxImZMXIo_MZOIlgweCgisspfAgYPsMm8yH5VWKtaaLJCuWPNxMDXLYMyhEZ3nw";

        // 1. Verify standard X.509 SPKI public key (as provided by IdPs / OpenSSL pubout)
        let cfg_spki = AuthConfig {
            jwt_public_key: Some(spki_pem.to_vec()),
            ..Default::default()
        };
        let claims = cfg_spki
            .verify(valid_token)
            .expect("SPKI key must successfully verify token");
        assert_eq!(claims.subject, "alice-rsa");
        assert_eq!(claims.roles, vec!["admin"]);

        // 2. Verify PKCS#1 RSA public key
        let cfg_pkcs1 = AuthConfig {
            jwt_public_key: Some(pkcs1_pem.to_vec()),
            ..Default::default()
        };
        let claims = cfg_pkcs1
            .verify(valid_token)
            .expect("PKCS#1 key must successfully verify token");
        assert_eq!(claims.subject, "alice-rsa");

        // 3. Verify corrupted token fails closed
        let tampered_token = format!("{valid_token}corrupted");
        assert!(cfg_spki.verify(&tampered_token).is_err());
    }
}
