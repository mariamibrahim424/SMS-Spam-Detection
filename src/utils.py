import matplotlib.pyplot as plt
import pandas as pd 
from wordcloud import WordCloud
from data_preprocessing import pre_process_data

# Plot class distribution (ham vs spam)
def plot_class_distribution(data,title='Distrubtion'):
    class_counts = data['label'].value_counts(normalize=True) * 100  # Normalize gives proportions, *100 to get percentage
    class_counts.plot(kind='bar', color=['green', 'red'])
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Percentage (%)')
    plt.xticks([0, 1], ['Ham', 'Spam'], rotation=0)
    plt.show()

def generate_wordcloud(clean_df, label='ham'):
    if label not in ['ham', 'spam']:
        raise ValueError("The label must be either 'ham' or 'spam'.")
    filtered_messages = clean_df[clean_df['label'] == label]['cleaned_sms']

    text = ' '.join(filtered_messages)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off the axis
    plt.title(f'Word Cloud for {label.capitalize()} Messages')
    plt.show()


# Main function that runs the program
def main():
    # Load and clean the data
    file_path = '../data/raw_sms.csv'  # Update the path to your dataset
    df = pre_process_data(file_path)
    
    # Plot the class distribution
    plot_class_distribution(df, title='SMS Class Distribution')
    
    # Generate word cloud for Ham messages
    generate_wordcloud(df, label='ham')
    
    # Generate word cloud for Spam messages
    generate_wordcloud(df, label='spam')

# Run the main function
if __name__ == "__main__":
    main()
