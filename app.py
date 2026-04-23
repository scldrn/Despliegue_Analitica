import os
import pickle
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
MPL_DIR = BASE_DIR / ".matplotlib"
HEART_CATEGORICAL = ["hypertension", "heart_disease", "ever_married", "smoking_status"]
REQUIRED_FILES = [
    "ataque_corazon.xlsx",
    "modelo_svm_completo.pkl",
    "modelo_svm_optimizado.joblib",
    "modelo-class.pkl",
    "videojuegos.csv",
    "videojuegos-datosFuturos.csv",
]

os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
MPL_DIR.mkdir(exist_ok=True)
warnings.filterwarnings("ignore")


def print_section(title):
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def check_required_files():
    missing = [name for name in REQUIRED_FILES if not (BASE_DIR / name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Faltan archivos requeridos: {missing_text}")


def prepare_heart_features(df, feature_names):
    x_data = df.drop(columns=["stroke_ataque_corazon"], errors="ignore").copy()

    for column in HEART_CATEGORICAL:
        if column in x_data.columns:
            x_data[column] = x_data[column].astype(str)

    x_data = pd.get_dummies(
        x_data,
        columns=[column for column in HEART_CATEGORICAL if column in x_data.columns],
        drop_first=True,
        dtype=int,
    )

    return x_data.reindex(columns=list(feature_names), fill_value=0)


def validate_heart_models():
    heart_df = pd.read_excel(BASE_DIR / "ataque_corazon.xlsx", sheet_name="Datos")

    with open(BASE_DIR / "modelo_svm_completo.pkl", "rb") as file:
        svm_bundle = pickle.load(file)

    svm_model = svm_bundle["model"]
    svm_feature_names = svm_bundle["feature_names"]
    svm_metrics = {key: float(value) for key, value in svm_bundle["metrics"].items()}

    x_heart = prepare_heart_features(heart_df, svm_feature_names)
    y_true = heart_df["stroke_ataque_corazon"].astype(str)
    y_true_num = svm_bundle["labelencoder"].transform(y_true)
    y_pred_num = svm_model.predict(x_heart)
    y_pred = svm_bundle["labelencoder"].inverse_transform(y_pred_num)
    y_proba = svm_model.predict_proba(x_heart)[:, 1]

    optimized_joblib = joblib.load(BASE_DIR / "modelo_svm_optimizado.joblib")
    x_optimized = prepare_heart_features(heart_df, optimized_joblib.feature_names_in_)
    optimized_pred = optimized_joblib.predict(x_optimized)

    with open(BASE_DIR / "modelo-class.pkl", "rb") as file:
        legacy_model, legacy_encoder, legacy_variables, legacy_scaler = pickle.load(file)

    x_legacy = prepare_heart_features(heart_df, legacy_variables).copy()
    x_legacy[["age", "avg_glucose_level"]] = legacy_scaler.transform(
        x_legacy[["age", "avg_glucose_level"]]
    )
    legacy_pred = legacy_encoder.inverse_transform(legacy_model.predict(x_legacy))

    resultados_svm = heart_df.assign(
        prediccion=y_pred,
        probabilidad_si=np.round(y_proba, 4),
    ).head(10)

    summary = pd.DataFrame(
        {
            "artefacto": [
                "modelo_svm_completo.pkl",
                "modelo_svm_optimizado.joblib",
                "modelo-class.pkl",
            ],
            "score": [
                round(accuracy_score(y_true, y_pred), 4),
                round(accuracy_score(y_true_num, optimized_pred), 4),
                round(accuracy_score(y_true, legacy_pred), 4),
            ],
            "comentario": [
                "Bundle principal con decoder y probabilidades",
                "Pipeline optimizado validado contra la etiqueta real",
                "Modelo legado validado con escalado solo en columnas numericas",
            ],
        }
    )

    return heart_df, svm_metrics, resultados_svm, summary


def build_regression_baseline():
    video_df = pd.read_csv(BASE_DIR / "videojuegos.csv")
    future_df = pd.read_csv(BASE_DIR / "videojuegos-datosFuturos.csv")

    x_data = video_df.drop(columns=["Presupuesto para invertir"])
    y_data = video_df["Presupuesto para invertir"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=0.2,
        random_state=42,
    )

    numeric_features = ["Edad"]
    categorical_features = [column for column in x_data.columns if column not in numeric_features]

    regression_model = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                            numeric_features,
                        ),
                        (
                            "cat",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            categorical_features,
                        ),
                    ]
                ),
            ),
            ("model", LinearRegression()),
        ]
    )

    regression_model.fit(x_train, y_train)
    test_pred = regression_model.predict(x_test)
    future_pred = regression_model.predict(future_df)

    metrics = pd.DataFrame(
        {
            "metrica": ["MAE", "RMSE", "R2"],
            "valor": [
                round(mean_absolute_error(y_test, test_pred), 2),
                round(float(np.sqrt(mean_squared_error(y_test, test_pred))), 2),
                round(r2_score(y_test, test_pred), 4),
            ],
        }
    )

    future_predictions = future_df.copy()
    future_predictions["prediccion_presupuesto"] = np.round(future_pred, 2)

    return metrics, future_predictions


def main():
    check_required_files()

    print_section("Validacion general")
    print(f"Directorio de trabajo: {BASE_DIR}")
    print("Archivos detectados:")
    files_df = pd.DataFrame({"archivo": sorted(path.name for path in BASE_DIR.iterdir() if path.is_file())})
    print(files_df.to_string(index=False))

    print_section("Clasificacion: ataque al corazon")
    heart_df, svm_metrics, resultados_svm, summary = validate_heart_models()
    print(f"Filas y columnas del dataset: {heart_df.shape}")
    print("Metricas guardadas en el artefacto principal:")
    print(svm_metrics)
    print("\nMuestra de predicciones:")
    print(resultados_svm.to_string(index=False))
    print("\nResumen de artefactos:")
    print(summary.to_string(index=False))

    print_section("Regresion: presupuesto en videojuegos")
    regression_metrics, future_predictions = build_regression_baseline()
    print("Metricas de regresion:")
    print(regression_metrics.to_string(index=False))
    print("\nPredicciones para datos futuros:")
    print(future_predictions.to_string(index=False))

    print_section("Estado final")
    print("Validacion completada sin errores.")
    print("app.py esta listo para ejecutarse con: python3 app.py")


if __name__ == "__main__":
    main()
