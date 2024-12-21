import pandas as pd


def process_numeric_columns(df, channel_specific_outliers=True):
    """
    Clean numeric columns by handling missing values and outliers.

    :param df: Input DataFrame
    :param channel_specific_outliers: If True, process outliers per channel
    :return: Cleaned DataFrame
    """
    numeric_columns = [
        'Likes',
        'Views',
        'Duration_minutes',
        'Click_rate',
        'Num_faces',
        'Brightness',
        'Contrast',
        'Saturation',
    ]

    def process_group(group):
        for column in numeric_columns:
            # Fill missing values with median
            median = group[column].median()
            group[column] = group[column].fillna(median)

            # Remove outliers using IQR
            Q1 = group[column].quantile(0.25)
            Q3 = group[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

            # Cap outliers
            group[column] = group[column].clip(
                lower=lower_bound, upper=upper_bound
            )
        return group

    if channel_specific_outliers:
        # Apply the group-wise processing without the 'Channel_name' column
        cleaned_df = df.groupby(
            'Channel_name',
            group_keys=False,
        ).apply(
            process_group,
            include_groups=False,
        )
    else:
        cleaned_df = process_group(df.copy())

    return cleaned_df


def clean_language(df):
    """
    Clean and standardize language column.

    :param df: Input DataFrame
    :return: DataFrame with cleaned language column
    """
    processed_groups = []

    for channel_name, group in df.groupby('Channel_name'):
        mode = group['Language'].mode()
        fill_value = mode.iloc[0] if not mode.empty else 'unknown'

        group['Language'] = group['Language'].fillna(fill_value)

        processed_groups.append(group)

    df = pd.concat(processed_groups)

    df['Language'] = df['Language'].str.lower()

    return df


def clean_text_columns(df):
    """
    Clean text-related columns.

    :param df: Input DataFrame
    :return: DataFrame with cleaned text columns
    """
    df['Text'] = df['Text'].fillna(''
                                  ).apply(lambda x: ' '.join(str(x).split()))
    return df


def clean_youtube_data(df, channel_specific_outliers=True):
    """
    Comprehensive data cleaning pipeline.

    :param df: Input DataFrame
    :param channel_specific_outliers: If True, process outliers per channel
    :return: Cleaned DataFrame
    """
    df = clean_language(df)
    df = clean_text_columns(df)
    df = process_numeric_columns(df, channel_specific_outliers)
    integer_columns = [
        'Duration_minutes',
        'Views',
        'Num_faces',
    ]

    for col in integer_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    return df


def main():
    data = pd.read_csv('storage/data/channel_videos_with_analysis.csv')
    cleaned_data = clean_youtube_data(data)
    cleaned_data.to_csv(
        'storage/cleaned_channel_videos_with_analysis.csv', index=False
    )


if __name__ == '__main__':
    main()
