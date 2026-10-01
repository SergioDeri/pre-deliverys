# PayFlow: visión general

PayFlow es la plataforma de cobros online de la empresa. Procesa pagos con tarjeta y transferencias para unos 3.000 comercios, con picos de 400 transacciones por segundo los viernes a la noche.

## Componentes

- **api-gateway**: NGINX delante de todo. Termina TLS, aplica rate limiting por comercio y enruta hacia los servicios internos.
- **pagos-api**: servicio en FastAPI que recibe las órdenes de pago, valida al comercio y escribe la transacción en estado `pendiente`.
- **pagos-worker**: consumidores en Python que leen la cola `pagos.autorizar`, hablan con el procesador de tarjetas y actualizan el estado de la transacción.
- **conciliacion**: job nocturno que cruza las transacciones del día con el archivo que manda el procesador.
- **notificaciones**: envía webhooks a los comercios cuando una transacción cambia de estado.

## Datos

La base principal es PostgreSQL 16 con una réplica de lectura. `pagos-api` y `pagos-worker` usan un pool de conexiones de 20 por instancia a través de PgBouncer. Redis guarda las claves de idempotencia durante 24 horas para que un comercio pueda reintentar una orden sin cobrar dos veces.

## Mensajería

RabbitMQ es el broker. Cada cola tiene una cola de mensajes muertos (`.dlq`) a la que van los mensajes después de cinco intentos fallidos. Nada se descarta en silencio: la DLQ se revisa a mano.

## Despliegue

Todo corre en Kubernetes. Los despliegues los hace Argo CD a partir del repositorio `payflow-deploy`; cada servicio tiene su propio `Application` y se puede volver a una revisión anterior sin tocar a los demás.
