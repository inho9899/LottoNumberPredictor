# 🎯 Lotto Number Predictor (Reinforcement Learning Based)

A PyTorch-based reinforcement learning model that predicts lottery numbers using historical data and LSTM neural networks with the REINFORCE algorithm.

## 📋 Project Overview

This project trains a policy model to generate remaining lotto numbers `[n2 ~ n7]` based on input number `n1`, using historical lotto data from all past draws. The model uses sequential decision-making to predict the most likely combinations while preventing duplicate number selection.

## 🔧 Tech Stack

- **Deep Learning Framework**: PyTorch
- **Neural Network Architecture**: LSTM-based Policy Network
- **Reinforcement Learning Algorithm**: REINFORCE Algorithm
- **Data Collection**: Web Scraping (BeautifulSoup)
- **Data Analysis**: Pandas, NumPy, Matplotlib
- **Special Techniques**: Action Masking for duplicate-free number selection

## 🎯 Key Features

- ✅ **Real Data Learning**: Training based on actual lotto winning number data
- ✅ **Sequential Generation**: Sequential decision-making for number prediction
- ✅ **Duplicate Prevention**: Action masking mechanism to prevent duplicate numbers
- ✅ **Stochastic Sampling**: Temperature-controlled sampling for diversity
- ✅ **Performance Analysis**: Comprehensive testing and visualization tools

## 📊 Model Architecture

```
Input (n1) → Embedding → LSTM → FC Layer → Softmax → Action (next number)
```

### Components:
1. **Embedding Layer**: Transforms lotto numbers (1-45) into high-dimensional vectors
2. **LSTM Layer**: Learns sequence patterns for next number prediction
3. **Fully Connected Layer**: Converts LSTM output to 45-class probabilities

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch pandas numpy matplotlib requests beautifulsoup4
```

### Installation

1. Clone this repository:
```bash
git clone https://github.com/inho9899/LottoNumberPredictor.git
cd lotto-predictor
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

1. **Data Collection**: Run the data collection cells to scrape historical lotto data
2. **Model Training**: Execute the training cells to train the REINFORCE policy
3. **Number Generation**: Use the trained model to generate lotto number combinations

```python
# Example usage
from lotto_predictor import generate_lotto_numbers

# Generate numbers starting with n1=25
result = generate_lotto_numbers(trained_policy, n1=25)
print(f"Generated numbers: {result}")
```

## 📁 Project Structure

```
lotto/
├── lotto.ipynb              # Main Jupyter notebook
├── README.md                # This file
├── lotto_data.csv           # Historical lotto data (generated)
├── lotto_policy_model.pth   # Trained model weights (generated)
└── requirements.txt         # Python dependencies
```

## 🧠 How It Works

### 1. Data Collection
- Scrapes historical lotto winning numbers from lottohell.com
- Collects data in format: `[n1, n2, n3, n4, n5, n6, bonus]`
- Handles errors and implements delays to prevent server overload

### 2. Policy Network
- **Input**: Current sequence of selected numbers
- **Processing**: LSTM processes the sequence to understand patterns
- **Output**: Probability distribution over remaining valid numbers

### 3. REINFORCE Training
- **Environment**: LottoEnv simulates the number selection process
- **Reward**: +1 for each number that matches actual winning numbers
- **Learning**: Policy gradient method optimizes selection strategy

### 4. Number Generation
- Given starting number `n1`, generates remaining 5 numbers
- Uses action masking to prevent duplicates
- Temperature scaling controls randomness vs. exploitation

## 📊 Performance Metrics

The model is evaluated on:
- **Accuracy**: Number of matches with actual winning numbers
- **Diversity**: Variation in generated combinations
- **Consistency**: Stability of results for same inputs

## 🔬 Model Limitations

- **Probabilistic Nature**: Lotto is inherently random, limiting prediction accuracy
- **Data Bias**: Historical patterns may not reflect future draws
- **Reward Design**: Simple matching reward may not capture full lotto dynamics

## 🚀 Future Improvements

### Model Architecture
- Implement Attention Mechanisms for better pattern recognition
- Explore Transformer architectures for sequence modeling
- Develop ensemble methods combining multiple models

### Training Methods
- Apply more stable algorithms like PPO or A2C
- Implement multi-task learning for different prize tiers
- Use transfer learning with other lottery datasets

### Data Enhancement
- Incorporate external factors (seasonality, events)
- Add statistical features (consecutiveness, odd-even ratios)
- Implement real-time data updates

## 📈 Results

The trained model achieves:
- Average reward of X.XX per episode
- XX% accuracy on validation data
- Diverse number generation with temperature scaling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**Important**: This model was created for educational and research purposes only. 

- Do not use this for actual lottery purchases
- Lottery outcomes are random and unpredictable
- Gambling can be addictive - please gamble responsibly
- The authors are not responsible for any losses incurred

## 🎓 Educational Value

This project serves as an excellent learning resource for:
- **Reinforcement Learning**: REINFORCE algorithm implementation
- **Sequential Decision Making**: LSTM-based policy networks
- **Web Scraping**: Data collection from web sources
- **Deep Learning**: PyTorch model development and training

