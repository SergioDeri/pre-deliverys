# Conciliación nocturna

## Qué hace

El job `conciliacion` corre todos los días a las 03:00 como un `CronJob` de Kubernetes. Cruza las transacciones `aprobada` del día anterior con el archivo de liquidación que manda Cobralia, el procesador de tarjetas, y marca cada una como `conciliada` o `con_diferencia`.

## El archivo de Cobralia

Cobralia deja el archivo por SFTP en `/liquidaciones/AAAAMMDD.csv` antes de las 02:30. Cada línea trae el identificador de autorización, el monto en centavos, la moneda y la comisión cobrada. Si a las 03:00 el archivo no está, el job reintenta cada 15 minutos hasta las 06:00 y después falla con el código `PF-7002`.

## Reglas de cruce

- Se cruza por identificador de autorización, nunca por monto: dos pagos del mismo comercio pueden tener el mismo monto.
- Una diferencia de monto de hasta 1 centavo se acepta como redondeo de la conversión de moneda.
- Una transacción aprobada en PayFlow que no figura en el archivo queda `con_diferencia` y se revisa a mano al día siguiente.

## Salida

El resultado se guarda en la tabla `conciliaciones` y se publica un resumen en el canal `#pagos-conciliacion`. Finanzas usa ese resumen para liberar los fondos a los comercios.
