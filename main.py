import pandas as pd
import numpy as np
import os
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

PERM_FILE = "data/input/Пермский край - Население.xlsx"
SVERDLOVSK_FILE = "data/input/Свердловская область - Население.xlsx"
OUTPUT_DIR = "data/output"
MODEL_REPORT = os.path.join(OUTPUT_DIR, "population_model_report.txt")

def load_and_combine_datasets():

    df_perm = pd.read_excel(PERM_FILE)
    print(f"Пермский край: {len(df_perm)} строк, {len(df_perm.columns)} колонок")
    
    df_sverd = pd.read_excel(SVERDLOVSK_FILE)
    print(f"Свердловская область: {len(df_sverd)} строк, {len(df_sverd.columns)} колонок")
    
    df_perm_std = standardize_perm_columns(df_perm)
    df_sverd_std = standardize_sverd_columns(df_sverd)
    
    combined_df = pd.concat([df_perm_std, df_sverd_std], ignore_index=True)
    print(f"Объединенный датасет: {len(combined_df)} строк")
    
    return combined_df

def standardize_perm_columns(df):
    result = df.copy()
    
    column_mapping = {
        'Longitude': 'lon',
        'Latitude': 'lat',
        'ЧН_Расчет': 'population',
        'Yandex add': 'address',
        'id': 'house_id'
    }
    
    result = result.rename(columns=column_mapping)
    
    if 'house_id' not in result.columns:
        result['house_id'] = range(1, len(result) + 1)
    
    result['region'] = 'Пермский край'
    
    return result

def standardize_sverd_columns(df):
    result = df.copy()
    
    column_mapping = {
        'LON': 'lon',
        'LAT': 'lat',
        'INHAB': 'population',
        'ADDRESS': 'address',
        'HOUSE_ID': 'house_id'
    }
    
    result = result.rename(columns=column_mapping)
    
    result['region'] = 'Свердловская область'
    
    return result

def parse_address_string(df):
    result = df.copy()
    
    result['parsed_country'] = ''
    result['parsed_region'] = ''
    result['parsed_city'] = ''
    result['parsed_street'] = ''
    result['parsed_house'] = ''
    
    for idx, row in result.iterrows():
        address = str(row['address'])
        
        if row['region'] == 'Пермский край':
            parts = [p.strip() for p in address.split(',')]
            if len(parts) >= 5:
                result.at[idx, 'parsed_country'] = parts[0] if len(parts) > 0 else ''
                result.at[idx, 'parsed_region'] = parts[1] if len(parts) > 1 else ''
                result.at[idx, 'parsed_city'] = parts[2] if len(parts) > 2 else ''
                result.at[idx, 'parsed_street'] = parts[3] if len(parts) > 3 else ''
                result.at[idx, 'parsed_house'] = parts[4] if len(parts) > 4 else ''
        
        elif row['region'] == 'Свердловская область':
            parts = [p.strip() for p in address.split(',')]
            if len(parts) >= 5:
                if 'обл.' in parts[0]:
                    result.at[idx, 'parsed_region'] = parts[0]
                else:
                    result.at[idx, 'parsed_country'] = parts[0]
                
                result.at[idx, 'parsed_city'] = parts[2] if len(parts) > 2 else parts[1]
                
                if len(parts) >= 4:
                    street_parts = []
                    house_parts = []
                    found_street = False
                    
                    for i in range(3, len(parts)):
                        part = parts[i]
                        if any(keyword in part.lower() for keyword in ['ул.', 'улица', 'пер.', 'переулок', 'д.', 'дом']):
                            if not found_street and 'д.' not in part.lower() and 'дом' not in part.lower():
                                street_parts.append(part)
                            else:
                                house_parts.append(part)
                                found_street = True
                        elif not found_street:
                            street_parts.append(part)
                        else:
                            house_parts.append(part)
                    
                    result.at[idx, 'parsed_street'] = ', '.join(street_parts)
                    result.at[idx, 'parsed_house'] = ', '.join(house_parts)
    
    result['house_number'] = result['parsed_house'].apply(extract_house_number)
    
    result['settlement_type'] = result['parsed_city'].apply(detect_settlement_type)
    
    print(f"Уникальных городов/поселений: {result['parsed_city'].nunique()}")
    print(f"Уникальных улиц: {result['parsed_street'].nunique()}")
    print(f"Примеры разобранных адресов:")
    for i in range(min(3, len(result))):
        print(f"  {result.iloc[i]['region']}: {result.iloc[i]['parsed_city']}, {result.iloc[i]['parsed_street']}")
    
    return result

def extract_house_number(house_str):
    if not isinstance(house_str, str):
        return 1
    
    import re
    numbers = re.findall(r'\d+', house_str)
    if numbers:
        return int(numbers[0])
    return 1

def detect_settlement_type(settlement):
    if not isinstance(settlement, str):
        return 'other'
    
    settlement_lower = settlement.lower()
    
    if any(word in settlement_lower for word in ['город', 'г.', 'г ', 'г.']):
        return 'city'
    elif any(word in settlement_lower for word in ['село', 'с.', 'с ', 'с.']):
        return 'village'
    elif any(word in settlement_lower for word in ['деревня', 'дер.', 'д.', 'д ']):
        return 'rural'
    elif any(word in settlement_lower for word in ['поселок', 'пос.', 'пгт', 'п.']):
        return 'township'
    elif any(word in settlement_lower for word in ['район', 'округ', 'р-н']):
        return 'district'
    elif any(word in settlement_lower for word in ['обл.', 'область']):
        return 'region'
    else:
        return 'other'

def create_features(df):
    result = df.copy()
    
    # Географические признаки
    result['coord_sum'] = result['lon'] + result['lat']
    result['coord_diff'] = result['lon'] - result['lat']
    
    # Региональные признаки
    le_region = LabelEncoder()
    result['region_encoded'] = le_region.fit_transform(result['region'])

    # Признаки из адреса
    le_settlement = LabelEncoder()
    result['settlement_type_encoded'] = le_settlement.fit_transform(result['settlement_type'].fillna('other'))
    
    # Длина названия улицы
    result['street_name_length'] = result['parsed_street'].apply(lambda x: len(str(x)))
    
    # Номер дома и его характеристики
    result['house_number'] = result['house_number'].fillna(1).astype(int)
    result['house_is_even'] = (result['house_number'] % 2 == 0).astype(int)
    result['house_category'] = pd.cut(
        result['house_number'],
        bins=[0, 10, 50, 100, float('inf')],
        labels=['small', 'medium', 'large', 'very_large']
    )
    
    # Кодируем категорию дома
    le_house_cat = LabelEncoder()
    result['house_category_encoded'] = le_house_cat.fit_transform(result['house_category'])
    
    # Признаки пространственной плотности
    result['lat_rounded'] = (result['lat'] * 100).round() / 100
    result['lon_rounded'] = (result['lon'] * 100).round() / 100
    
    # Плотность зданий в ячейке
    density = result.groupby(['lat_rounded', 'lon_rounded']).size()
    density_dict = density.to_dict()
    result['density'] = result.apply(
        lambda row: density_dict.get((row['lat_rounded'], row['lon_rounded']), 1),
        axis=1
    )
    
    # Оценочная площадь здания
    result['estimated_area'] = 50 + result['house_number'] * 2 + result['density'] * 10
    result['estimated_area'] = result['estimated_area'].clip(lower=30, upper=5000)
    
    # Признаки типа здания
    result['likely_residential'] = result['density'].apply(lambda x: 1 if x > 5 else 0)
    
    # Дополнительные признаки из Свердловской области
    if 'LEVELS' in result.columns:
        result['building_levels'] = pd.to_numeric(result['LEVELS'], errors='coerce').fillna(1)
    else:
        result['building_levels'] = 1
    
    if 'AREA' in result.columns:
        result['building_area'] = pd.to_numeric(result['AREA'], errors='coerce').fillna(result['estimated_area'])
    else:
        result['building_area'] = result['estimated_area']
    
    result = result.drop(['lat_rounded', 'lon_rounded'], axis=1, errors='ignore')
    
    print(f"Создано признаков: {len([col for col in result.columns if 'encoded' in col or 'parsed_' in col])}")
    print(f"Общее количество колонок: {len(result.columns)}")
    
    return result

def prepare_training_data(df):
    base_features = [
        'lon', 'lat',                   
        'coord_sum', 'coord_diff',       
        'region_encoded',                
        'settlement_type_encoded',       
        'street_name_length',            
        'house_number',                  
        'house_is_even',                
        'house_category_encoded',       
        'density',                       
        'estimated_area',               
        'building_levels',               
        'building_area',                 
        'likely_residential'             
    ]
    
    available_features = [col for col in base_features if col in df.columns]
    
    print(f"Используемые признаки ({len(available_features)}):")
    for feature in available_features:
        print(f"  - {feature}")
    
    target_column = 'population'
    
    if target_column not in df.columns:
        print(f"ОШИБКА: Целевая колонка '{target_column}' не найдена")
        return None, None, None
    
    X = df[available_features].copy()
    y = df[target_column].copy()
    
    X = X.fillna(0)
    
    y = y.clip(lower=0.1) 
    
    print(f"Размер X: {X.shape}")
    print(f"Размер y: {y.shape}")
    print(f"Диапазон целевой переменной: {y.min():.2f} - {y.max():.2f}")
    print(f"Среднее значение населения: {y.mean():.2f}")
    print(f"Медианное значение населения: {y.median():.2f}")
    
    return X, y, available_features

def train_and_evaluate_model(X, y, features):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"Тренировочная выборка: {X_train.shape[0]} записей")
    print(f"Тестовая выборка: {X_test.shape[0]} записей")
    
    # Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучение модели, алгоритм Случайный лес
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    metrics = {
        'train': {
            'mae': mean_absolute_error(y_train, y_pred_train),
            'mse': mean_squared_error(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'r2': r2_score(y_train, y_pred_train)
        },
        'test': {
            'mae': mean_absolute_error(y_test, y_pred_test),
            'mse': mean_squared_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'r2': r2_score(y_test, y_pred_test)
        }
    }
    
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nРезультаты обучения модели:")
    print("Тренировочная выборка:")
    print(f"  MAE: {metrics['train']['mae']:.2f}")
    print(f"  RMSE: {metrics['train']['rmse']:.2f}")
    print(f"  R²: {metrics['train']['r2']:.4f}")
    
    print("\nТестовая выборка:")
    print(f"  MAE: {metrics['test']['mae']:.2f}")
    print(f"  RMSE: {metrics['test']['rmse']:.2f}")
    print(f"  R²: {metrics['test']['r2']:.4f}")
    
    return model, scaler, metrics, feature_importance

def analyze_model_performance(model, X_test_scaled, y_test, y_pred_test, df_test):
   
    results_df = df_test.copy()
    results_df['actual'] = y_test.values
    results_df['predicted'] = y_pred_test
    
    results_df['error'] = results_df['predicted'] - results_df['actual']
    results_df['abs_error'] = np.abs(results_df['error'])
    results_df['error_percentage'] = (results_df['abs_error'] / results_df['actual']) * 100
    
    results_df['error_percentage'] = results_df['error_percentage'].replace([np.inf, -np.inf], np.nan)
    
    # Статистика ошибок
    print("Статистика ошибок прогнозирования:")
    print(f"  Средняя абсолютная ошибка: {results_df['abs_error'].mean():.2f}")
    print(f"  Медианная абсолютная ошибка: {results_df['abs_error'].median():.2f}")
    print(f"  Максимальная ошибка: {results_df['abs_error'].max():.2f}")
    
    if results_df['error_percentage'].notna().any():
        valid_percentages = results_df['error_percentage'].dropna()
        print(f"  Средняя процентная ошибка: {valid_percentages.mean():.1f}%")
        print(f"  Медианная процентная ошибка: {valid_percentages.median():.1f}%")
    
    if 'region' in results_df.columns:
        print("\nОшибки по регионам:")
        for region in results_df['region'].unique():
            if pd.notna(region):
                subset = results_df[results_df['region'] == region]
                if len(subset) > 0:
                    mae = subset['abs_error'].mean()
                    count = len(subset)
                    print(f"  {region}: MAE={mae:.2f}, записей={count}")
    
    if 'settlement_type' in results_df.columns:
        print("\nОшибки по типам населенных пунктов:")
        for stype in results_df['settlement_type'].unique():
            if pd.notna(stype):
                subset = results_df[results_df['settlement_type'] == stype]
                if len(subset) > 0:
                    mae = subset['abs_error'].mean()
                    count = len(subset)
                    print(f"  {stype}: MAE={mae:.2f}, записей={count}")
    
    return results_df

def save_model_report(model, scaler, metrics, feature_importance, results_df):
  
    report_content = []
    report_content.append("ОТЧЕТ О МОДЕЛИ ОЦЕНКИ ЧИСЛЕННОСТИ НАСЕЛЕНИЯ")
    report_content.append("")
    
    report_content.append("1. ИНФОРМАЦИЯ О ДАННЫХ")
    report_content.append("-" * 40)
    report_content.append(f"Общее количество записей: {len(results_df) * 1.25:.0f}")
    report_content.append(f"Из Пермского края: {len(results_df[results_df['region'] == 'Пермский край']) * 1.25:.0f}")
    report_content.append(f"Из Свердловской области: {len(results_df[results_df['region'] == 'Свердловская область']) * 1.25:.0f}")
    report_content.append(f"Количество признаков: {len(feature_importance)}")
    report_content.append(f"Диапазон целевой переменной: {results_df['actual'].min():.2f} - {results_df['actual'].max():.2f}")
    report_content.append("")
    
    report_content.append("2. МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ")
    report_content.append("-" * 40)
    report_content.append("Тренировочная выборка:")
    report_content.append(f"  MAE: {metrics['train']['mae']:.2f}")
    report_content.append(f"  RMSE: {metrics['train']['rmse']:.2f}")
    report_content.append(f"  R²: {metrics['train']['r2']:.4f}")
    report_content.append("")
    report_content.append("Тестовая выборка:")
    report_content.append(f"  MAE: {metrics['test']['mae']:.2f}")
    report_content.append(f"  RMSE: {metrics['test']['rmse']:.2f}")
    report_content.append(f"  R²: {metrics['test']['r2']:.4f}")
    report_content.append("")
    
    report_content.append("3. АНАЛИЗ ОШИБОК")
    report_content.append("-" * 40)
    report_content.append(f"Средняя абсолютная ошибка (тест): {results_df['abs_error'].mean():.2f}")
    report_content.append(f"Медианная абсолютная ошибка: {results_df['abs_error'].median():.2f}")
    report_content.append(f"Максимальная ошибка: {results_df['abs_error'].max():.2f}")
    
    if results_df['error_percentage'].notna().any():
        valid_percentages = results_df['error_percentage'].dropna()
        report_content.append(f"Средняя процентная ошибка: {valid_percentages.mean():.1f}%")
        report_content.append(f"Медианная процентная ошибка: {valid_percentages.median():.1f}%")
    report_content.append("")
    
    if 'region' in results_df.columns:
        report_content.append("4. ОШИБКИ ПО РЕГИОНАМ")
        report_content.append("-" * 40)
        for region in results_df['region'].unique():
            if pd.notna(region):
                subset = results_df[results_df['region'] == region]
                if len(subset) > 0:
                    mae = subset['abs_error'].mean()
                    count = len(subset)
                    report_content.append(f"{region:25} MAE={mae:.2f}, записей={count}")
        report_content.append("")
    
    with open(MODEL_REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
    
    print(f"  Отчет сохранен в: {MODEL_REPORT}")
    
    import joblib
    model_path = os.path.join(OUTPUT_DIR, 'population_model_combined.joblib')
    scaler_path = os.path.join(OUTPUT_DIR, 'scaler_combined.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"  Модель сохранена в: {model_path}")
    print(f"  Scaler сохранен в: {scaler_path}")
    
    results_path = os.path.join(OUTPUT_DIR, 'combined_population_predictions.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8')
    print(f"  Подробные результаты сохранены в: {results_path}")

def main():
    print("МОДЕЛЬ ОЦЕНКИ ЧИСЛЕННОСТИ НАСЕЛЕНИЯ")
    print("Обучение на данных Пермского края и Свердловской области")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n1. ЗАГРУЗКА И ОБЪЕДИНЕНИЕ ДАННЫХ")
    df = load_and_combine_datasets()
    
    if df is None or len(df) == 0:
        print("ОШИБКА: Не удалось загрузить данные")
        return
    
    print("\n2. ПАРСИНГ АДРЕСОВ")
    df = parse_address_string(df)
    
    print("\n3. СОЗДАНИЕ ПРИЗНАКОВ")
    df = create_features(df)
    
    print("\n4. ПОДГОТОВКА ДЛЯ ОБУЧЕНИЯ")
    X, y, features = prepare_training_data(df)
    
    if X is None:
        print("ОШИБКА: Не удалось подготовить данные для обучения")
        return
    
    print("\n5. ОБУЧЕНИЕ МОДЕЛИ")
    model, scaler, metrics, feature_importance = train_and_evaluate_model(X, y, features)
    
    print("\n6. АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ")
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    X_test_scaled = scaler.transform(X_test)
    y_pred_test = model.predict(X_test_scaled)
    
    _, test_indices = train_test_split(df.index, test_size=0.2, random_state=42, shuffle=True)
    df_test = df.loc[test_indices].copy()
    
    results_df = analyze_model_performance(model, X_test_scaled, y_test, y_pred_test, df_test)
    
    print("\n7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    save_model_report(model, scaler, metrics, feature_importance, results_df)
    
    print("МОДЕЛЬ УСПЕШНО ОБУЧЕНА НА ОБОИХ РЕГИОНАХ")
    print(f"Общий размер датасета: {len(df)} записей")
    print(f"Точность модели (R² на тесте): {metrics['test']['r2']:.4f}")
    print(f"Средняя ошибка (MAE): {metrics['test']['mae']:.2f}")

    print(f"\nДля применения модели к новым данным используйте:")
    print(f"model = joblib.load('{os.path.join(OUTPUT_DIR, 'population_model_combined.joblib')}')")

if __name__ == "__main__":
    main()