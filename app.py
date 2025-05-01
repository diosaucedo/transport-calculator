
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import base64
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

FIXED_COST = 2607

# === Model Training ===
def load_and_train_model():
    final_quarterly_path = 'Final_Quarterly.xlsx'
    merged_final_path = 'merged_final.xlsx'

    quarter_df = pd.read_excel(final_quarterly_path)
    merged_df = pd.read_excel(merged_final_path)

    merged_df = merged_df[['Location', 'PROGRAM', '% Donated (per location)']]
    program_means = merged_df.groupby('PROGRAM')['% Donated (per location)'].mean().reset_index()
    program_means.rename(columns={'% Donated (per location)': 'Program_Mean_Donated_%'}, inplace=True)
    quarter_df = quarter_df.merge(merged_df, on=['Location', 'PROGRAM'], how='left')
    quarter_df = quarter_df.merge(program_means, on='PROGRAM', how='left')

    quarter_df['Estimated_Donated_%'] = (
        0.8 * quarter_df['% Donated (per location)'] + 0.2 * quarter_df['Program_Mean_Donated_%']
    )

    programs = ['AGENCY', 'SP', 'BP', 'MP', 'PP']
    df = quarter_df[quarter_df['PROGRAM'].isin(programs) & (quarter_df['Cost'] > 1)].copy()
    df['Weight_Tier'] = 'Unassigned'
    df['Total_Weight'] = df['Weight']

    df.loc[(df['PROGRAM'] == 'AGENCY') & (df['Total_Weight'] < 8000), 'Weight_Tier'] = 'Low'
    df.loc[(df['PROGRAM'] == 'AGENCY') & (df['Total_Weight'] >= 8000) & (df['Total_Weight'] <= 20000), 'Weight_Tier'] = 'Mid'
    df.loc[(df['PROGRAM'] == 'AGENCY') & (df['Total_Weight'] > 20000), 'Weight_Tier'] = 'High'
    df.loc[(df['PROGRAM'] == 'BP') & (df['Total_Weight'] < 7311), 'Weight_Tier'] = 'All'
    df.loc[(df['PROGRAM'] == 'MP') & (df['Total_Weight'] < 6000), 'Weight_Tier'] = 'Low'
    df.loc[(df['PROGRAM'] == 'MP') & (df['Total_Weight'] >= 6000), 'Weight_Tier'] = 'High'
    df.loc[(df['PROGRAM'] == 'SP') & (df['Total_Weight'] < 2000), 'Weight_Tier'] = 'Low'
    df.loc[(df['PROGRAM'] == 'SP') & (df['Total_Weight'] >= 2000), 'Weight_Tier'] = 'High'
    df.loc[(df['PROGRAM'] == 'PP') & (df['Total_Weight'] < 150000), 'Weight_Tier'] = 'All'
    df = df[df['Weight_Tier'] != 'Unassigned']

    features = ['Region', 'PROGRAM', 'Quarter', 'hh', 'Weight', 'Estimated_Donated_%']
    target = 'Cost'

    model_dict = {}
    for (program, tier), subset in df.groupby(['PROGRAM', 'Weight_Tier']):
        if len(subset) < 20: continue
        X = subset[features]
        y = np.log1p(subset[target])
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        pre = ColumnTransformer([
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['Region', 'PROGRAM', 'Quarter']),
            ('num', 'passthrough', ['hh', 'Weight', 'Estimated_Donated_%'])
        ])

        pipe = Pipeline([
            ('preprocessor', pre),
            ('regressor', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200))
        ])

        pipe.fit(X_train, y_train)
        model_dict[(program, tier)] = pipe

    return model_dict

model_dict = load_and_train_model()

# === Forecast Function ===
def forecast_cost_per_lb_mile(program, region, hh, total_weight, donated_pct, miles):
    donated_ratio = donated_pct / 100.0
    quarterly_total = total_weight / 4

    if program == 'AGENCY':
        tier = 'Low' if quarterly_total < 8000 else 'Mid' if quarterly_total <= 20000 else 'High'
    elif program == 'MP':
        tier = 'Low' if quarterly_total < 6000 else 'High'
    elif program == 'SP':
        tier = 'Low' if quarterly_total < 2000 else 'High'
    elif program in ['BP', 'PP']:
        tier = 'All'
    else:
        return None, None

    model = model_dict.get((program, tier))
    if model is None: return None, None

    input_df = pd.DataFrame([{
        'Region': region,
        'PROGRAM': program,
        'Quarter': 'Q1',
        'hh': hh,
        'Weight': total_weight,
        'Estimated_Donated_%': donated_ratio
    }])

    pred = model.predict(input_df)[0]
    base_cost = np.expm1(pred) * 4
    total_cost = base_cost + FIXED_COST
    cost_per_lb_per_mile = total_cost / (total_weight * miles)
    return cost_per_lb_per_mile, total_cost

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        weight = float(request.form["weight"])
        distance = float(request.form["distance"])
        agency = request.form["agency"]
        hh = int(request.form["hh"])
        donated = float(request.form["donated"])
        region_label = request.form["region_label"]

        cost_per_lb, total = forecast_cost_per_lb_mile(agency, region_label, hh, weight, donated, distance)
        result = {
            "agency": agency,
            "region": region_label,
            "weight": weight,
            "hh": hh,
            "donated": donated,
            "distance": distance,
            "cost_per_lb": round(cost_per_lb, 4),
            "total": round(total, 2)
        }
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
