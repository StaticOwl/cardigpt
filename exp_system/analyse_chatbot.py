import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('exp_system/score_report.csv')
sns.set(style="whitegrid")
# ["Question", "Response", "Metric 1-Coverage Score", "Metric 2 - Grounding Score", "Metric 3 - Coherence Score", "Final Score"]
final_score_avg=df["Final Score"].mean()
 
plt.figure(figsize=(10, 6))
 
ax=df.plot(kind='bar', x='Question', y=["Metric 1-Coverage Score", "Metric 2 - Grounding Score", "Metric 3 - Coherence Score", 'Final Score'], figsize=(12, 8), legend=True)
ax.axhline(final_score_avg, color='b', linestyle=':', linewidth=5,label=f'Average Final Score: {final_score_avg:.2f}') 
plt.xticks(rotation=90)
 
plt.title('Comparison of Metrics for Each Question')
plt.xlabel('Questions')
plt.ylabel('Scores')
plt.legend()
plt.tight_layout()

os.makedirs("exp_system/chatbot_plots",exist_ok=True)
plt.savefig("exp_system/chatbot_plots/comparison.png",format="png")
plt.show()
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[["Metric 1-Coverage Score", "Metric 2 - Grounding Score", "Metric 3 - Coherence Score", 'Final Score']])

plt.title('Score Distribution of Each Metric')
plt.ylabel('Scores')
plt.savefig("exp_system/chatbot_plots/distribution.png",format="png")
plt.show()
plt.figure(figsize=(8, 6))
correlation_matrix = df[["Metric 1-Coverage Score", "Metric 2 - Grounding Score", "Metric 3 - Coherence Score", 'Final Score']].corr()
 
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Metrics')
plt.savefig("exp_system/chatbot_plots/correlation.png",format="png")
plt.show()