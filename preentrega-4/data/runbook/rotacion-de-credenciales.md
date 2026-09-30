# Runbook: rotación de credenciales

## Credenciales del procesador de tarjetas

Se rotan cada 90 días o ante cualquier sospecha de filtración. La nueva credencial se carga en el secreto `procesador-credenciales` de Kubernetes con una clave nueva, sin borrar la anterior.

Después se reinicia `pagos-worker` con `kubectl rollout restart`. Durante el reinicio conviven workers con la credencial vieja y con la nueva; el procesador acepta las dos durante 24 horas.

## Usuario de base de datos

El usuario de aplicación de PostgreSQL se rota creando uno nuevo con los mismos permisos, actualizando PgBouncer y recién después eliminando el anterior. Nunca se cambia la contraseña del usuario en uso: las conexiones abiertas empezarían a fallar al reconectar.
