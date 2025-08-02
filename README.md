# ICBC Vehicle-Prediction 

By exploring vehicle population data published by ICBC on Tableau Public, we developed a pipeline that involved data cleaning, data visualization, and data analysis to predict vehicle type. 

## Relevancy and Business Problem Addressed 

**Question**: "What vehicle characteristics (e.g., Make, Model Year, Body Style, Anti Theft Device Indicator) most strongly predict whether a vehicle is electric?" 

**Relevancy**: 

<p> ICBC is always evaluating risk models for insurance and sustainability adaptation. </p>

<p> By identifying which vehicle characteristics (such as Make, Model Year, Body Style, and Anti-Theft Device Indicator) are strong predictors of electric vehicle ownership, ICBC can enhance its actuarial/prediction models for risk evaluation and premium pricing. </p>

<p> The insights we developed also support long-term sustainability efforts that ICBC can adopt based on forecasting EV adoption trends across different regions, enabling ICBC to align with provincial climate initiatives and inform targeted incentive programs. </p>

<p>Additionally, our analysis helps in resource allocation for claims processing and vehicle servicing, particularly in adapting to the unique repair and operational process/needs of EVs. These findings can also be used to simulate future adoption scenarios, aiding ICBC in strategic forecasting, regulatory planning, and data-driven policy development in the future.</p>


## Dependencies

OS: Windows (For PowerBI compatibility) 

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required libraries:

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scipy
pip install scikit-learn
pip install tabulate
```

## Usage

To see the results of any statistical tests (relevant p-values) and figures generated, run `Data_Analysis.py`. Download and open the `.pbix' file on Power Bi to view the dashboard that showcases our results more clearly. 

## Data Visualizations 

**1.)** *Dataset Description*: 

<img width="1113" height="612" alt="Dataset descriotion" src="https://github.com/user-attachments/assets/5c75ee18-0f6c-4b66-862e-567bb7d93388" />

**2.)** *Dashboard* 

- Users are able to filter by vehicle type

<img width="1120" height="626" alt="Dashboard" src="https://github.com/user-attachments/assets/ae6d11aa-7300-44e7-af04-275e6957a9c7" />

**3.)** *Results*

- Summary of accuracy/AUC score
- Confusion matrix for classifcation
- ROC curve comparison based on different supervised classification models tested
  
<img width="1122" height="628" alt="Results" src="https://github.com/user-attachments/assets/f2a4ff3a-a0fe-4caa-a2e1-2dc42e1a509b" />


## Contributors

<ul>
<li>Abdullah Naeem</li>
<li>Akshay Wadhwa </li>
</ul>

