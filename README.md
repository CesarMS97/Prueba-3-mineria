# API BernoulliNB — Cesar

Plantilla lista para que tu BernoulliNB se conecte al **dashboard grupal AutoInsight**
de Exequiel: <https://api-autoinsight.vercel.app>

## Lo que tenés que hacer

### 1. Poner tus 2 artefactos en `models/`

```
models/
├── bernoulli_nb.joblib              # el estimador entrenado
└── bernoulli_nb_metadata.joblib     # dict con features, accuracy, etc.
```

**Forma esperada de `bernoulli_nb_metadata.joblib`:**

```python
{
    "features": ["segmento_corto", "vel_alta", "orient_ns", "orient_eo",
                 "es_horario_peligroso", "es_fin_semana"],        # las binarias que usaste
    "classes":  ["no_critica", "critica"],                        # del LabelEncoder
    "target":   "es_zona_critica_nocturna",
    "accuracy":         0.918,
    "accuracy_cv":      0.912,
    "recall":           0.86,           # recall de la clase positiva
    "precision":        0.84,
    "f1":               0.85,
    "confusion_matrix": [[18244, 612], [402, 2250]],
}
```

Si tu metadata tiene otros nombres (`labels` en vez de `classes`, etc.) andá a [app/main.py:38](app/main.py#L38) y ajustá. **No toques el resto del archivo.**

### 2. Correr local para probar

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Abrir <http://localhost:8000/docs> → Swagger te deja testear todos los endpoints.

### 3. Deploy

**Con Docker (recomendado):**
```bash
docker build -t cesar-bnb-api .
docker run -p 8000:8000 cesar-bnb-api
```

**En Easypanel / Render / Fly.io:** subí el repo, apunta al Dockerfile y listo.

### 4. Verificar contra el dashboard

Una vez deployado, abrir el dashboard con tu URL como `?api=`:

```
https://api-autoinsight.vercel.app/dashboard?api=https://tu-api-bnb.tu-dominio.cl
```

El banner debería decir:
> 🟢 **Detectado: Clasificación Bayesiana** · BernoulliNB / GaussianNB

Y aparecen automáticamente los paneles:
- ✅ Clases del modelo (crítica / no crítica)
- ✅ Probabilidades P(feature \| clase)
- ✅ Métricas + matriz confusión
- Ocultos: PCA 2D, Distribución por cluster, Pipeline K-Means (esos son del K-Means de Exequiel, no aplican a vos)

## Qué endpoints expone esta plantilla

| Método | Path | Para qué |
|---|---|---|
| `GET` | `/` | Info del servicio |
| `GET` | `/health` | Healthcheck |
| `GET` | `/classes` | Lista de clases con conteos y color → **panel Clases** |
| `GET` | `/probabilities` | Matriz `P(feature=1\|clase)` por cada feature → **panel Probabilidades** |
| `GET` | `/metrics` | Accuracy, recall, precision, F1, matriz → **panel Métricas** |
| `POST` | `/predict` | Predice clase + probabilidades → **formulario** |
| `GET` | `/docs` | Swagger UI auto-generado |
| `GET` | `/openapi.json` | Esquema OpenAPI → **panel Discovery** |

## El endpoint estrella: `/probabilities`

Este es **el panel que distingue a BernoulliNB** del Random Forest de Cesar. Devuelve la matriz de probabilidades condicionales `P(feature=1 \| clase)` que el modelo aprendió durante el entrenamiento. El dashboard la pinta como tabla con cells coloreadas — diferencias grandes entre `P(feature\|crítica)` vs `P(feature\|no_crítica)` significan que esa feature **discrimina bien**.

Por ejemplo: si `P(es_horario_peligroso=1 \| crítica) = 0.99` pero `P(es_horario_peligroso=1 \| no_critica) = 0.05`, esa feature por sí sola ya distingue casi perfectamente.

Esta vista no la puede mostrar el Random Forest porque sus probabilidades vienen de votación de árboles, no de una distribución parametrizable.

## Si tu modelo usa otras features

El método `construir_features()` en [app/main.py:100](app/main.py#L100) ya deriva todas las binarias típicas del proyecto. El DataFrame final filtra automáticamente solo las que están en tu `metadata["features"]`. Si tu modelo usa alguna que no está acá, agregala al diccionario `valores` (línea ~120).

## Soporte

Si el dashboard no detecta tu API o algún panel sale en blanco, lo más probable es:
1. **CORS bloqueado**: chequeá que el CORSMiddleware (línea 70) esté abierto a `*` o al menos a `https://api-autoinsight.vercel.app`.
2. **Falla algún endpoint**: abrí `/docs` y testealo manual — Swagger te dice el error exacto.
3. **`feature_log_prob_` no existe**: tu modelo no es BernoulliNB sino otro (Gaussian, Multinomial). Avisame y adapto el endpoint.
4. **Forma de respuesta distinta**: comparar con [API_CONTRACT.md](../../API_CONTRACT.md) en la raíz del repo grupal.
