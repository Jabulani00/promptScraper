from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
import os
import re
import json
from urllib.parse import urlparse, urljoin
import time
import random
from datetime import datetime
try:
    from googlesearch import search as google_search
except ImportError:
    print("Google search package not installed. Internet search will be disabled.")
    google_search = None

# Create the Flask application
app = Flask(__name__)

# Create templates directory if it doesn't exist
if not os.path.exists('templates'):
    os.makedirs('templates')

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

def get_meta_tags(soup):
    """Extract meta tags from the page"""
    meta_tags = {}
    for tag in soup.find_all('meta'):
        if tag.get('name'):
            meta_tags[tag.get('name')] = tag.get('content')
        elif tag.get('property'):
            meta_tags[tag.get('property')] = tag.get('content')
    return meta_tags

def get_headers(soup):
    """Extract headers (h1-h6) from the page"""
    headers = {}
    for i in range(1, 7):
        header_tags = soup.find_all(f'h{i}')
        if header_tags:
            headers[f'h{i}'] = [tag.text.strip() for tag in header_tags]
    return headers

def detect_technologies(html_content, url):
    """Simple detection of technologies used on the website"""
    technologies = []
    domain = urlparse(url).netloc
    
    # Check for common libraries and frameworks
    tech_signatures = {
        'jQuery': 'jquery',
        'Bootstrap': 'bootstrap',
        'React': 'react',
        'Vue.js': ['vue.js', 'vue.min.js'],
        'Angular': 'angular',
        'WordPress': 'wp-content',
        'Font Awesome': 'font-awesome',
        'Google Analytics': ['google-analytics.com', 'ga.js', 'analytics.js'],
        'Google Tag Manager': 'googletagmanager.com',
        'Google Fonts': 'fonts.googleapis.com',
        'Cloudflare': 'cloudflare',
        'Shopify': 'shopify',
        'Wix': 'wix.com',
        'Squarespace': 'squarespace'
    }
    
    for tech, signatures in tech_signatures.items():
        if not isinstance(signatures, list):
            signatures = [signatures]
        for signature in signatures:
            if signature.lower() in html_content.lower():
                technologies.append(tech)
                break
    
    return list(set(technologies))

def extract_domain_info(url):
    """Extract basic domain information"""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    
    try:
        import whois
        domain_info = whois.whois(domain)
        return {
            'domain': domain,
            'registrar': domain_info.registrar,
            'creation_date': str(domain_info.creation_date) if domain_info.creation_date else None,
            'expiration_date': str(domain_info.expiration_date) if domain_info.expiration_date else None,
            'name_servers': domain_info.name_servers if hasattr(domain_info, 'name_servers') else None
        }
    except:
        return {
            'domain': domain,
            'error': 'Could not retrieve domain information'
        }

def fix_relative_urls(base_url, url):
    """Convert relative URLs to absolute URLs"""
    if not url:
        return ''
    
    # Check if URL is already absolute
    if url.startswith(('http://', 'https://', '//')):
        return url
    
    # Handle data URLs
    if url.startswith('data:'):
        return url
    
    # Fix relative URLs
    return urljoin(base_url, url)

@app.route('/scrape', methods=['POST'])
def scrape():
    """Handle the scraping request for a specific URL"""
    # Get URL from the form
    url = request.form.get('url')
    
    # Get search terms if provided
    search_term = request.form.get('search_term', '')
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    try:
        # Add http:// if not present
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        # Send request to get the page content
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Initialize result dictionary
        result = {
            'title': soup.title.text if soup.title else 'No title found',
            'paragraphs': [],
            'links': [],
            'images': [],
            'meta_tags': get_meta_tags(soup),
            'headers': get_headers(soup),
            'technologies': detect_technologies(response.text, url),
            'domain_info': extract_domain_info(url)
        }
        
        # Extract paragraphs
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            if search_term:
                if search_term.lower() in p.text.lower():
                    result['paragraphs'].append(p.text.strip())
            else:
                result['paragraphs'].append(p.text.strip())
        
        # Extract links
        links = soup.find_all('a', href=True)
        for link in links:
            absolute_url = fix_relative_urls(url, link.get('href', ''))
            
            if search_term:
                if search_term.lower() in link.text.lower():
                    result['links'].append({
                        'text': link.text.strip() or link.get('href', ''),
                        'href': absolute_url
                    })
            else:
                result['links'].append({
                    'text': link.text.strip() or link.get('href', ''),
                    'href': absolute_url
                })
        
        # Extract images
        images = soup.find_all('img')
        for img in images:
            img_src = img.get('src', '')
            if img_src:
                absolute_src = fix_relative_urls(url, img_src)
                
                if img.get('alt') and search_term:
                    if search_term.lower() in img.get('alt', '').lower():
                        result['images'].append({
                            'src': absolute_src,
                            'alt': img.get('alt', 'No alt text'),
                            'width': img.get('width', ''),
                            'height': img.get('height', '')
                        })
                else:
                    result['images'].append({
                        'src': absolute_src,
                        'alt': img.get('alt', 'No alt text'),
                        'width': img.get('width', ''),
                        'height': img.get('height', '')
                    })
        
        # Get forms
        forms = soup.find_all('form')
        result['forms'] = []
        for form in forms:
            form_data = {
                'action': form.get('action', ''),
                'method': form.get('method', 'get'),
                'fields': []
            }
            
            for input_tag in form.find_all(['input', 'textarea', 'select']):
                field = {
                    'type': input_tag.name if input_tag.name != 'input' else input_tag.get('type', 'text'),
                    'name': input_tag.get('name', ''),
                    'id': input_tag.get('id', '')
                }
                form_data['fields'].append(field)
            
            result['forms'].append(form_data)
        
        # Extract scripts
        scripts = soup.find_all('script', src=True)
        result['scripts'] = [fix_relative_urls(url, script.get('src', '')) for script in scripts]
        
        # Extract stylesheets
        stylesheets = soup.find_all('link', rel='stylesheet')
        result['stylesheets'] = [fix_relative_urls(url, css.get('href', '')) for css in stylesheets]
        
        # Count the extracted items
        result['stats'] = {
            'paragraph_count': len(result['paragraphs']),
            'link_count': len(result['links']),
            'image_count': len(result['images']),
            'form_count': len(result['forms']),
            'script_count': len(result['scripts']),
            'stylesheet_count': len(result['stylesheets'])
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    """Handle internet search requests"""
    # Get the search query
    query = request.form.get('query')
    num_results = int(request.form.get('num_results', 5))
    
    if not query:
        return jsonify({'error': 'Search query is required'}), 400
    
    if not google_search:
        return jsonify({'error': 'Internet search functionality is not available. Please install the required package.'}), 500
    
    try:
        # Perform the search
        search_results = []
        for url in google_search(query, num_results=num_results):
            search_results.append({'url': url})
            
        # Get more details about each result
        for i, result in enumerate(search_results):
            try:
                # Add a small delay to avoid rate limiting
                if i > 0:
                    time.sleep(random.uniform(1, 3))
                    
                url = result['url']
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=5)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract basic info
                result['title'] = soup.title.text if soup.title else 'No title found'
                
                # Extract meta description
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                result['description'] = meta_desc.get('content', '') if meta_desc else ''
                
                # Extract a snippet of text
                paragraphs = soup.find_all('p')
                text_chunks = [p.text.strip() for p in paragraphs if len(p.text.strip()) > 50]
                
                if text_chunks:
                    # Find paragraph that best matches the query
                    best_match = max(text_chunks, key=lambda x: sum(q.lower() in x.lower() for q in query.split()))
                    result['snippet'] = best_match[:300] + '...' if len(best_match) > 300 else best_match
                else:
                    result['snippet'] = 'No content available'
                    
            except Exception as e:
                result['error'] = str(e)
                result['title'] = result.get('title', url)
                result['description'] = result.get('description', 'Could not fetch details')
                result['snippet'] = result.get('snippet', 'Could not fetch content')
        
        return jsonify({
            'query': query,
            'results': search_results,
            'count': len(search_results),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True)






    from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import io
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # For flash messages
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# File extensions we'll allow
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

def read_file(file):
    """Read CSV or Excel file into pandas DataFrame"""
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    if filename.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:  # Excel file
        df = pd.read_excel(file_path)
    
    # Clean up temp file
    os.remove(file_path)
    return df

def generate_summary_stats(df):
    """Generate summary statistics for DataFrame"""
    # Basic info
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        summary = df[numeric_cols].describe().T
        summary['missing'] = df[numeric_cols].isnull().sum()
        summary['missing_pct'] = (df[numeric_cols].isnull().sum() / len(df)) * 100
        summary = summary.round(2)
        return summary.to_html(classes="table table-striped table-hover")
    else:
        return "<p>No numeric columns found for summary statistics.</p>"

def detect_outliers(df):
    """Detect outliers in numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return "<p>No numeric columns found for outlier detection.</p>"
    
    # Create subplots for boxplots
    fig = make_subplots(rows=len(numeric_cols), cols=1, 
                        subplot_titles=numeric_cols,
                        vertical_spacing=0.05)
    
    for i, col in enumerate(numeric_cols, 1):
        fig.add_trace(go.Box(y=df[col], name=col), row=i, col=1)
    
    fig.update_layout(height=300 * len(numeric_cols), showlegend=False, 
                     title_text="Outlier Analysis with Box Plots")
    
    return fig.to_html(full_html=False, include_plotlyjs=False)

def generate_correlation_plot(df):
    """Generate correlation heatmap for numeric columns"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        return "<p>Not enough numeric columns for correlation analysis.</p>"
    
    corr = numeric_df.corr()
    
    # Create correlation heatmap
    fig = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.columns,
                    colorscale='RdBu_r',
                    zmin=-1, zmax=1))
    
    fig.update_layout(
        title='Correlation Heatmap',
        height=600,
        width=800)
    
    return fig.to_html(full_html=False, include_plotlyjs=False)

def compare_datasets(df1, df2):
    """Compare two datasets and highlight differences"""
    # Check for common columns
    common_cols = list(set(df1.columns) & set(df2.columns))
    
    if not common_cols:
        return "<p>No common columns found to compare datasets.</p>"
    
    # For numeric columns, compare distributions
    numeric_cols = [col for col in common_cols if col in df1.select_dtypes(include=[np.number]).columns
                   and col in df2.select_dtypes(include=[np.number]).columns]
    
    comparison_html = "<h3>Common Columns: " + ", ".join(common_cols) + "</h3>"
    
    if numeric_cols:
        fig = make_subplots(rows=len(numeric_cols), cols=1, 
                           subplot_titles=numeric_cols,
                           vertical_spacing=0.1)
        
        for i, col in enumerate(numeric_cols, 1):
            fig.add_trace(go.Histogram(x=df1[col], name='File 1', opacity=0.7), row=i, col=1)
            fig.add_trace(go.Histogram(x=df2[col], name='File 2', opacity=0.7), row=i, col=1)
        
        fig.update_layout(height=400 * len(numeric_cols), barmode='overlay',
                        title_text="Distribution Comparison for Numeric Columns")
        
        comparison_html += fig.to_html(full_html=False, include_plotlyjs=False)
    
    # Generate statistics comparison table
    stats_comparison = {
        'Number of Rows': [len(df1), len(df2)],
        'Number of Columns': [len(df1.columns), len(df2.columns)],
        'Common Columns': [len(common_cols), len(common_cols)],
        'Missing Values': [df1[common_cols].isnull().sum().sum(), df2[common_cols].isnull().sum().sum()]
    }
    
    stats_df = pd.DataFrame(stats_comparison, index=['File 1', 'File 2'])
    comparison_html += "<h3>Dataset Statistics Comparison</h3>"
    comparison_html += stats_df.to_html(classes="table table-striped")
    
    return comparison_html

def generate_visualizations(df):
    """Generate various visualizations based on data"""
    visualizations = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 1:
        return visualizations
    
    # 1. Distribution plot for first numeric column
    if len(numeric_cols) > 0:
        col = numeric_cols[0]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[col], nbinsx=30))
        fig.update_layout(title=f"Distribution of {col}", 
                         xaxis_title=col, 
                         yaxis_title="Count")
        
        visualizations.append({
            'title': f"Distribution of {col}",
            'plot': fig.to_html(full_html=False, include_plotlyjs=False),
            'description': f"Histogram showing the frequency distribution of {col} values."
        })
    
    # 2. Scatter plot for first two numeric columns if available
    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[0], numeric_cols[1]
        fig = px.scatter(df, x=col1, y=col2)
        fig.update_layout(title=f"Relationship between {col1} and {col2}")
        
        visualizations.append({
            'title': f"Scatter Plot: {col1} vs {col2}",
            'plot': fig.to_html(full_html=False, include_plotlyjs=False),
            'description': f"Scatter plot showing the relationship between {col1} and {col2}."
        })
    
    # 3. Bar chart for first categorical column if available
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        col = categorical_cols[0]
        value_counts = df[col].value_counts().nlargest(10)  # Top 10 categories
        fig = px.bar(x=value_counts.index, y=value_counts.values)
        fig.update_layout(title=f"Top Categories in {col}",
                         xaxis_title=col,
                         yaxis_title="Count")
        
        visualizations.append({
            'title': f"Bar Chart: Top Categories in {col}",
            'plot': fig.to_html(full_html=False, include_plotlyjs=False),
            'description': f"Bar chart showing the distribution of top categories in {col}."
        })
    
    return visualizations

def generate_insights(df1, df2=None):
    """Generate insights about the data"""
    insights = []
    
    # Basic insights for df1
    insights.append(f"Dataset 1 contains {df1.shape[0]} rows and {df1.shape[1]} columns.")
    
    # Missing values
    missing = df1.isnull().sum().sum()
    if missing > 0:
        missing_pct = (missing / (df1.shape[0] * df1.shape[1])) * 100
        insights.append(f"Dataset 1 has {missing} missing values ({missing_pct:.2f}% of all data).")
    else:
        insights.append("Dataset 1 has no missing values.")
    
    # Numeric columns
    numeric_cols = df1.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        insights.append(f"Dataset 1 has {len(numeric_cols)} numeric columns: {', '.join(numeric_cols)}.")
        
        # Check for correlation
        if len(numeric_cols) >= 2:
            corr = df1[numeric_cols].corr()
            # Get the top correlation pair
            corr_values = corr.unstack()
            corr_values = corr_values[corr_values < 1.0]  # Remove self-correlations
            if not corr_values.empty:
                max_corr = corr_values.abs().max()
                if max_corr > 0.7:
                    max_pair = corr_values.abs().idxmax()
                    insights.append(f"Strong correlation detected between {max_pair[0]} and {max_pair[1]} (r={corr_values.loc[max_pair]:.2f}).")
    
    # Categorical columns
    cat_cols = df1.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        insights.append(f"Dataset 1 has {len(cat_cols)} categorical columns: {', '.join(cat_cols)}.")
        
        # Check for cardinality
        for col in cat_cols:
            unique_values = df1[col].nunique()
            if unique_values == 1:
                insights.append(f"Column '{col}' has only one unique value and might not be useful for analysis.")
            elif unique_values > 10 and unique_values / len(df1) > 0.9:
                insights.append(f"Column '{col}' has high cardinality ({unique_values} values) and might be an ID column.")
    
    # Compare datasets if df2 is provided
    if df2 is not None:
        common_cols = list(set(df1.columns) & set(df2.columns))
        insights.append(f"The two datasets share {len(common_cols)} common columns.")
        
        # Row count comparison
        row_diff = abs(len(df1) - len(df2))
        row_diff_pct = (row_diff / max(len(df1), len(df2))) * 100
        if row_diff > 0:
            insights.append(f"The datasets differ by {row_diff} rows ({row_diff_pct:.2f}% difference).")
        
        # Compare common numeric columns
        common_numeric = [col for col in common_cols if col in df1.select_dtypes(include=[np.number]).columns
                         and col in df2.select_dtypes(include=[np.number]).columns]
        
        if common_numeric:
            for col in common_numeric:
                mean_diff = abs(df1[col].mean() - df2[col].mean())
                max_mean = max(abs(df1[col].mean()), abs(df2[col].mean()))
                if max_mean > 0:
                    mean_diff_pct = (mean_diff / max_mean) * 100
                    if mean_diff_pct > 10:  # If means differ by more than 10%
                        insights.append(f"The mean values for '{col}' differ by {mean_diff_pct:.2f}% between datasets.")
    
    return insights

@app.route('/upload_data', methods=['POST'])
def upload_data():
    if 'file1' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file1 = request.files['file1']
    if file1.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if not allowed_file(file1.filename):
        flash('Invalid file type. Please upload CSV or Excel files.')
        return redirect(request.url)
    
    try:
        # Read the first file
        df1 = read_file(file1)
        df1_html = df1.head(10).to_html(classes="table table-striped", index=False)
        summary1_html = generate_summary_stats(df1)
        
        # Check if a second file was uploaded
        df2 = None
        df2_html = None
        summary2_html = None
        comparison_results = None
        
        if 'file2' in request.files and request.files['file2'].filename != '':
            file2 = request.files['file2']
            if allowed_file(file2.filename):
                df2 = read_file(file2)
                df2_html = df2.head(10).to_html(classes="table table-striped", index=False)
                summary2_html = generate_summary_stats(df2)
                comparison_results = compare_datasets(df1, df2)
        
        # Generate visualizations
        visualizations = generate_visualizations(df1)
        
        # Generate correlation plot if requested
        correlation_plot = None
        if 'correlation' in request.form:
            correlation_plot = generate_correlation_plot(df1)
        
        # Detect outliers if requested
        outliers = None
        if 'outlier_detection' in request.form:
            outliers = detect_outliers(df1)
        
        # Generate insights
        insights = generate_insights(df1, df2)
        
        return render_template('visualization.html', 
                              analysis_results=True,
                              df1_html=df1_html,
                              df2_html=df2_html,
                              summary1_html=summary1_html,
                              summary2_html=summary2_html,
                              visualizations=visualizations,
                              correlation_plot=correlation_plot,
                              outliers=outliers,
                              comparison_results=comparison_results,
                              insights=insights)
    
    except Exception as e:
        flash(f'Error processing file: {str(e)}')
        return redirect(url_for('visualization'))

if __name__ == '__main__':
    app.run(debug=True)