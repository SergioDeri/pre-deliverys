# Runbook: rollback de un deploy

## Cuándo hacerlo

Si después de un deploy suben los errores 5xx, la latencia p99 o las transacciones rechazadas, primero se vuelve atrás y después se investiga.

## Pasos

1. En Argo CD, abrir la `Application` del servicio afectado y ver el historial.
2. Elegir la revisión anterior y hacer `Rollback`. Argo CD desactiva el auto-sync de esa aplicación mientras dure.
3. Confirmar que las réplicas nuevas estén `Ready` y que las métricas vuelvan a la normalidad en 10 minutos.
4. Abrir un revert en `payflow-deploy` para que el repositorio coincida con lo que está corriendo, y reactivar el auto-sync.

## Migraciones de base de datos

Si el deploy incluía una migración, el rollback del código no la revierte. Las migraciones tienen que ser compatibles hacia atrás justamente para que este paso sea seguro.
