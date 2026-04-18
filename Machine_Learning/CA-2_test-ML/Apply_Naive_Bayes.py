file_path = '/home/pankaj/Downloads/study_material.pdf/ML/ML_CA_dataset/selected/titanic.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully from: {file_path}")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure the path and filename are correct.")
    print("Attempting to load Titanic dataset from a common URL...")
    try:
        df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
        print("Successfully loaded dataset from URL.")
    except Exception as e:
        print(f"Could not load dataset from URL either: {e}")
        df = pd.DataFrame()
if not df.empty:
    print("Original DataFrame head:")
    display(df.head())
    
    
    
    
    if 'Age' in df.columns:
        median_age = df['Age'].median()
        df['Age'].fillna(median_age, inplace=True)
        print(f"'Age' column imputed with median: {median_age}")
    else:
        print("'Age' column not found.")

if 'Cabin' in df.columns:
    df.drop('Cabin', axis=1, inplace=True)
    print("'Cabin' column dropped.")
else:
    print("'Cabin' column not found.")

if 'Embarked' in df.columns:
    mode_embarked = df['Embarked'].mode()[0]
    df['Embarked'].fillna(mode_embarked, inplace=True)
    print(f"'Embarked' column imputed with mode: {mode_embarked}")
else:
    print("'Embarked' column not found.")

print("\nDataFrame after imputation and dropping 'Cabin':")
display(df.head())