"""
Interactive House Price Predictor
Use this script to make predictions on your own house data!
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import os

print("="*60)
print("INTERACTIVE HOUSE PRICE PREDICTOR")
print("="*60)

# Check if trained model exists
model_file = "house_price_model.pkl"
if not os.path.exists(model_file):
    print("\nTraining model from house_data.csv...")
    # Load and train model
    data = pd.read_csv("house_data.csv")
    
    # Encode location
    le = LabelEncoder()
    data['location_encoded'] = le.fit_transform(data['location'])
    
    X = data[["area", "bedrooms", "bathrooms", "location_encoded", "age", "parking"]]
    y = data["price"]
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Save the model
    with open(model_file, 'wb') as f:
        pickle.dump({'model': model, 'location_encoder': le}, f)
    print("✓ Model trained and saved!")
else:
    print("\nLoading pre-trained model...")
    with open(model_file, 'rb') as f:
        saved_data = pickle.load(f)
        model = saved_data['model']
        le = saved_data['location_encoder']
    print("✓ Model loaded!")

location_map = {loc: idx for idx, loc in enumerate(le.classes_)}
print(f"\nAvailable Locations: {list(location_map.keys())}")

# Interactive prediction
def predict_house_price():
    print("\n" + "-"*60)
    print("Enter house details to predict price:")
    print("-"*60)
    
    try:
        area = float(input("Area (sq ft): "))
        bedrooms = int(input("Bedrooms: "))
        bathrooms = int(input("Bathrooms: "))
        location = input(f"Location ({'/'.join(location_map.keys())}): ").strip()
        age = int(input("Age (years): "))
        parking = int(input("Parking spaces: "))
        
        if location not in location_map:
            print(f"❌ Invalid location. Choose from: {list(location_map.keys())}")
            return
        
        location_encoded = location_map[location]
        
        # Make prediction
        house_features = [[area, bedrooms, bathrooms, location_encoded, age, parking]]
        predicted_price = model.predict(house_features)[0]
        
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"House Details:")
        print(f"  Area: {area:,.0f} sq ft")
        print(f"  Bedrooms: {bedrooms}")
        print(f"  Bathrooms: {bathrooms}")
        print(f"  Location: {location}")
        print(f"  Age: {age} years")
        print(f"  Parking: {parking} spaces")
        print(f"\n🏠 Predicted Price: ${predicted_price:,.2f}")
        print("="*60)
        
    except ValueError:
        print("❌ Invalid input. Please enter numbers where required.")
    except Exception as e:
        print(f"❌ Error: {e}")

# Main loop
while True:
    print("\nOptions:")
    print("1. Predict house price")
    print("2. Predict multiple houses (batch)")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        predict_house_price()
    
    elif choice == '2':
        print("\nBatch Prediction")
        print("-"*60)
        csv_file = input("Enter CSV filename (with header: area,bedrooms,bathrooms,location,age,parking): ").strip()
        
        try:
            batch_data = pd.read_csv(csv_file)
            batch_data['location_encoded'] = batch_data['location'].map(location_map)
            
            if batch_data['location_encoded'].isnull().any():
                print("❌ Invalid location found in CSV")
                continue
            
            X_batch = batch_data[["area", "bedrooms", "bathrooms", "location_encoded", "age", "parking"]]
            predictions = model.predict(X_batch)
            
            batch_data['predicted_price'] = predictions
            output_file = csv_file.replace('.csv', '_predictions.csv')
            batch_data.to_csv(output_file, index=False)
            
            print(f"\n✓ Predictions saved to: {output_file}")
            print(f"\nSample Results (first 5):")
            print(batch_data[['area', 'bedrooms', 'bathrooms', 'location', 'predicted_price']].head())
            
        except FileNotFoundError:
            print(f"❌ File not found: {csv_file}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif choice == '3':
        print("\nThank you for using the House Price Predictor!")
        break
    
    else:
        print("❌ Invalid option. Please select 1, 2, or 3.")
