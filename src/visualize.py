from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


class YouTubeDataAnalyzer:
    def __init__(self, filepath: str):
        self.df = self._load_data(filepath)
        self._preprocess_data()

    def _load_data(self, filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath)

    def _preprocess_data(self) -> None:
        self.df['Published_date'] = pd.to_datetime(self.df['Published_date'])
        self.df['Vectorized_text'] = self.df['Vectorized_text'].apply(eval)
        self.df['Vectorized_title'] = self.df['Vectorized_title'].apply(eval)

    def get_summary_statistics(self) -> Dict[str, Any]:
        numeric_columns = [
            'Subscribers',
            'Duration_minutes',
            'Likes',
            'Views',
            'Click_rate',
            'Num_faces',
        ]
        return {
            'description': self.df[numeric_columns].describe(),
            'info': self.df.info()
        }

    def visualize_data(self) -> None:
        plt.figure(figsize=(20, 15))

        # Views Distribution
        plt.subplot(2, 3, 1)
        sns.histplot(self.df['Views'], kde=True)
        plt.title('Distribution of Video Views')
        plt.xlabel('Views')

        # Correlation Heatmap
        plt.subplot(2, 3, 2)
        numeric_columns = [
            'Subscribers',
            'Duration_minutes',
            'Likes',
            'Views',
            'Click_rate',
            'Num_faces',
        ]
        correlation_matrix = self.df[numeric_columns].corr()
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            linewidths=0.5,
        )
        plt.title('Correlation Heatmap')

        # Duration and Likes Box Plot
        plt.subplot(2, 3, 3)
        self.df.boxplot(column=['Duration_minutes', 'Likes'])
        plt.title('Box Plot: Duration and Likes')

        # Views vs Likes Scatter Plot
        plt.subplot(2, 3, 4)
        plt.scatter(self.df['Views'], self.df['Likes'])
        plt.title('Views vs Likes')
        plt.xlabel('Views')
        plt.ylabel('Likes')

        # Videos per Language Bar Plot
        plt.subplot(2, 3, 5)
        self.df['Language'].value_counts().plot(kind='bar')
        plt.title('Videos per Language')
        plt.xlabel('Language')
        plt.ylabel('Number of Videos')

        plt.tight_layout()
        plt.show()

    def perform_statistical_analysis(self) -> Dict[str, Any]:
        # Pearson Correlation
        views_likes_correlation = self.df['Views'].corr(self.df['Likes'])

        # T-Test for Likes based on Number of Faces
        likes_multiple_faces = self.df[self.df['Num_faces'] > 1]['Likes']
        likes_single_face = self.df[self.df['Num_faces'] <= 1]['Likes']
        t_statistic, p_value = stats.ttest_ind(
            likes_multiple_faces,
            likes_single_face,
        )

        return {
            'views_likes_correlation': views_likes_correlation,
            't_test': {
                't_statistic': t_statistic,
                'p_value': p_value,
            }
        }

    def analyze_time_series(self) -> None:
        daily_metrics = self.df.groupby(self.df['Published_date'].dt.date).agg(
            {
                'Views': 'mean',
                'Likes': 'mean',
                'Duration_minutes': 'mean',
            }
        ).reset_index()

        plt.figure(figsize=(15, 5))

        metrics = ['Views', 'Likes', 'Duration_minutes']
        titles = [
            'Daily Average Views',
            'Daily Average Likes',
            'Daily Average Duration',
        ]

        for i, (metric, title) in enumerate(zip(metrics, titles), 1):
            plt.subplot(1, 3, i)
            plt.plot(daily_metrics['Published_date'], daily_metrics[metric])
            plt.title(title)
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def ml_readiness_check(self) -> Dict[str, Any]:
        return {
            'missing_values':
                self.df.isnull().sum(),
            'data_types':
                self.df.dtypes,
            'potential_features':
                [
                    'Duration_minutes',
                    'Likes',
                    'Views',
                    'Click_rate',
                    'Num_faces',
                ]
        }


def main():
    # Usage example
    analyzer = YouTubeDataAnalyzer(
        'storage/cleaned_channel_videos_with_analysis.csv'
    )

    # Run various analyses
    print("Summary Statistics:")
    summary_stats = analyzer.get_summary_statistics()

    print("\nStatistical Analysis:")
    stat_analysis = analyzer.perform_statistical_analysis()
    print(
        f"Views-Likes Correlation: {stat_analysis['views_likes_correlation']}"
    )
    print(f"T-Test Results: {stat_analysis['t_test']}")

    print("\nML Readiness Check:")
    ml_readiness = analyzer.ml_readiness_check()
    print("Potential Features:", ml_readiness['potential_features'])

    # Visualizations (uncomment to run)
    analyzer.visualize_data()
    analyzer.analyze_time_series()


if __name__ == "__main__":
    main()
