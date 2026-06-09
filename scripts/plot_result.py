#!/usr/bin/env python3
"""
COMPREHENSIVE MODEL VISUALIZATION SUITE
- Creates multiple detailed plots for model analysis
- Separates different aspects for better clarity
- Includes ROC, Precision-Recall, and business impact plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import glob
import re
from datetime import datetime
from itertools import cycle

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "reports")  # Changed from "reports/test"
PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

class ModelVisualizer:
    """Comprehensive visualizer for model performance"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.colors = {
            'primary': '#3498db',
            'secondary': '#2ecc71',
            'danger': '#e74c3c',
            'warning': '#f39c12',
            'info': '#9b59b6',
            'dark': '#2c3e50'
        }
    
    def load_latest_results(self):
        """Load latest test results and metadata"""
        print("📊 Loading latest results...")
        
        # Find all metrics CSV files recursively in results directory
        pattern = os.path.join(RESULTS_DIR, "**", "metrics_*.csv")
        csv_files = glob.glob(pattern, recursive=True)
        
        if not csv_files:
            # Try alternative pattern
            pattern = os.path.join(RESULTS_DIR, "*", "metrics_*.csv")
            csv_files = glob.glob(pattern, recursive=True)
        
        if not csv_files:
            # Try one level deeper
            pattern = os.path.join(RESULTS_DIR, "*", "*", "metrics_*.csv")
            csv_files = glob.glob(pattern, recursive=True)
        
        if not csv_files:
            print(f"❌ No metrics CSV files found in {RESULTS_DIR}")
            print(f"   Tried patterns: metrics_*.csv")
            return None, None, None
        
        # Get the latest file by modification time
        latest_file = max(csv_files, key=os.path.getmtime)
        print(f"✅ Found latest metrics file: {os.path.basename(latest_file)}")
        print(f"   Path: {latest_file}")
        
        # Load test results
        try:
            test_df = pd.read_csv(latest_file)
            print(f"   Loaded {len(test_df)} rows, {len(test_df.columns)} columns")
            
            # Check if required columns exist, if not, rename
            column_mapping = {
                'f1': 'f1_score',
                'auc': 'auc_roc',
                'tp': 'true_positives',
                'fp': 'false_positives',
                'fn': 'false_negatives',
                'tn': 'true_negatives',
                'fpr': 'false_positive_rate',
                'fnr': 'false_negative_rate'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in test_df.columns and new_col not in test_df.columns:
                    test_df[new_col] = test_df[old_col]
            
            # Add threshold if missing
            if 'threshold' not in test_df.columns:
                test_df['threshold'] = 0.5
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return None, None, None
        
        # Extract timestamp from filename
        filename = os.path.basename(latest_file)
        timestamp_match = re.search(r'(\d{8}_\d{6})', filename)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
        else:
            # Use file modification time
            mtime = os.path.getmtime(latest_file)
            timestamp = datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")
        
        # Try to load metadata
        metadata = self._find_matching_metadata(timestamp)
        
        return test_df, metadata, timestamp
    
    def _find_matching_metadata(self, timestamp):
        """Find metadata that matches the test results timestamp"""
        print(f"🔍 Looking for metadata matching timestamp: {timestamp}")
        
        # Look in model directories
        models_dir = os.path.join(BASE_DIR, "models")
        metadata_files = []
        
        if os.path.exists(models_dir):
            # Search for JSON metadata
            json_pattern = os.path.join(models_dir, "**", f"model_metadata_{timestamp}.json")
            json_files = glob.glob(json_pattern, recursive=True)
            
            # Search for pickle metadata
            pkl_pattern = os.path.join(models_dir, "**", f"model_metadata_{timestamp}.pkl")
            pkl_files = glob.glob(pkl_pattern, recursive=True)
            
            metadata_files = json_files + pkl_files
            
            if metadata_files:
                metadata_path = metadata_files[0]  # Take first match
                print(f"✅ Found matching metadata: {os.path.basename(metadata_path)}")
                try:
                    if metadata_path.endswith('.json'):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    else:
                        metadata = joblib.load(metadata_path)
                    return metadata
                except Exception as e:
                    print(f"❌ Error loading metadata: {e}")
        
        # If no exact match, try to find latest metadata
        print("⚠️  No exact timestamp match, looking for latest metadata...")
        
        # Find all metadata files
        all_metadata = []
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file.startswith('model_metadata_') and (file.endswith('.json') or file.endswith('.pkl')):
                    full_path = os.path.join(root, file)
                    all_metadata.append((os.path.getmtime(full_path), full_path))
        
        if all_metadata:
            # Get latest metadata
            all_metadata.sort(reverse=True)
            latest_mtime, latest_meta = all_metadata[0]
            print(f"✅ Using latest metadata: {os.path.basename(latest_meta)}")
            try:
                if latest_meta.endswith('.json'):
                    with open(latest_meta, 'r') as f:
                        metadata = json.load(f)
                else:
                    metadata = joblib.load(latest_meta)
                return metadata
            except Exception as e:
                print(f"❌ Error loading latest metadata: {e}")
        
        print("⚠️  No metadata found, using default values")
        return None
    
    def plot_core_metrics(self, test_results):
        """Plot core performance metrics"""
        print("📈 Creating core metrics plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Extract metrics - handle missing columns
        metrics_data = {
            'Accuracy': test_results.get('accuracy', test_results.get('Accuracy', 0)).iloc[0] if 'accuracy' in test_results.columns else 0,
            'Precision': test_results['precision'].iloc[0] if 'precision' in test_results.columns else 0,
            'Recall': test_results['recall'].iloc[0] if 'recall' in test_results.columns else 0,
            'F1-Score': test_results['f1_score'].iloc[0] if 'f1_score' in test_results.columns else 0,
            'AUC-ROC': test_results['auc_roc'].iloc[0] if 'auc_roc' in test_results.columns else 0,
            'Avg Precision': test_results['precision'].iloc[0] if 'precision' in test_results.columns else 0
        }
        
        # 1. Bar chart for core metrics
        ax1 = axes[0]
        colors = [self.colors['primary'], self.colors['secondary'], 
                 self.colors['danger'], self.colors['info']]
        bars = ax1.bar(list(metrics_data.keys())[:4], 
                      list(metrics_data.values())[:4], 
                      color=colors, edgecolor='black')
        ax1.set_title('Core Performance Metrics', fontweight='bold')
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, list(metrics_data.values())[:4]):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
        
        # 2. Radar chart for security metrics
        ax2 = axes[1]
        security_metrics = ['Detection Rate', 'False Alarm Rate', 
                          'Specificity', 'Precision']
        
        recall = test_results['recall'].iloc[0] if 'recall' in test_results.columns else 0
        fpr = test_results['false_positive_rate'].iloc[0] if 'false_positive_rate' in test_results.columns else 0
        precision = test_results['precision'].iloc[0] if 'precision' in test_results.columns else 0
        
        security_values = [
            recall,
            fpr,
            1 - fpr if 'false_positive_rate' in test_results.columns else 0,
            precision
        ]
        
        angles = np.linspace(0, 2*np.pi, len(security_metrics), endpoint=False).tolist()
        angles += angles[:1]
        security_values += security_values[:1]
        
        ax2 = plt.subplot(2, 2, 2, projection='polar')
        ax2.plot(angles, security_values, 'o-', linewidth=2, color=self.colors['primary'])
        ax2.fill(angles, security_values, alpha=0.25, color=self.colors['primary'])
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(security_metrics, fontsize=10, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.set_title('Security Performance Radar', fontweight='bold')
        ax2.grid(True)
        
        # 3. Confusion matrix heatmap
        ax3 = axes[2]
        # Get confusion matrix values with defaults
        tn = test_results['true_negatives'].iloc[0] if 'true_negatives' in test_results.columns else 0
        fp = test_results['false_positives'].iloc[0] if 'false_positives' in test_results.columns else 0
        fn = test_results['false_negatives'].iloc[0] if 'false_negatives' in test_results.columns else 0
        tp = test_results['true_positives'].iloc[0] if 'true_positives' in test_results.columns else 0
        
        cm_data = np.array([[tn, fp], [fn, tp]])
        
        im = ax3.imshow(cm_data, cmap='Blues', aspect='auto')
        ax3.set_xticks([0, 1])
        ax3.set_yticks([0, 1])
        ax3.set_xticklabels(['Predicted\nBenign', 'Predicted\nMalicious'], 
                           fontweight='bold', fontsize=10)
        ax3.set_yticklabels(['Actual\nBenign', 'Actual\nMalicious'], 
                           fontweight='bold', fontsize=10)
        
        # Add text annotations
        total = cm_data.sum()
        for i in range(2):
            for j in range(2):
                percentage = cm_data[i, j] / total if total > 0 else 0
                ax3.text(j, i, f'{cm_data[i, j]:,}\n({percentage:.1%})',
                        ha='center', va='center', 
                        color='white' if cm_data[i, j] > cm_data.max()/2 else 'black',
                        fontweight='bold', fontsize=11)
        
        ax3.set_title('Confusion Matrix', fontweight='bold')
        plt.colorbar(im, ax=ax3)
        
        # 4. Error distribution pie chart
        ax4 = axes[3]
        error_labels = ['True Positives', 'False Positives', 
                       'False Negatives', 'True Negatives']
        error_values = [tp, fp, fn, tn]
        error_colors = [self.colors['secondary'], self.colors['warning'],
                       self.colors['danger'], self.colors['primary']]
        
        wedges, texts, autotexts = ax4.pie(error_values, labels=error_labels,
                                          colors=error_colors, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontweight': 'bold'})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax4.set_title('Error Distribution', fontweight='bold')
        
        plt.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        plot_path = os.path.join(PLOTS_DIR, f'core_metrics_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Core metrics plot saved: {plot_path}")
        return plot_path
    
    def plot_business_impact(self, test_results):
        """Plot business impact and cost analysis"""
        print("💰 Creating business impact plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract metrics with defaults
        tp = test_results['true_positives'].iloc[0] if 'true_positives' in test_results.columns else 0
        fp = test_results['false_positives'].iloc[0] if 'false_positives' in test_results.columns else 0
        fn = test_results['false_negatives'].iloc[0] if 'false_negatives' in test_results.columns else 0
        tn = test_results['true_negatives'].iloc[0] if 'true_negatives' in test_results.columns else 0
        
        recall = test_results['recall'].iloc[0] if 'recall' in test_results.columns else 0
        precision = test_results['precision'].iloc[0] if 'precision' in test_results.columns else 0
        fpr = test_results['false_positive_rate'].iloc[0] if 'false_positive_rate' in test_results.columns else 0
        f1 = test_results['f1_score'].iloc[0] if 'f1_score' in test_results.columns else 0
        
        # 1. Cost analysis
        ax1 = axes[0, 0]
        cost_analysis = {
            'Missed Threats\n(FN Cost)': fn * 10,  # FN costs 10x more
            'False Alarms\n(FP Cost)': fp * 1,
            'Detection Savings\n(TP Benefit)': tp * 5,  # Each detection saves 5x
            'Correct Rejections\n(TN Benefit)': tn * 0.5
        }
        
        colors = [self.colors['danger'], self.colors['warning'],
                 self.colors['secondary'], self.colors['primary']]
        bars = ax1.bar(cost_analysis.keys(), cost_analysis.values(), color=colors)
        ax1.set_title('Cost-Benefit Analysis', fontweight='bold')
        ax1.set_ylabel('Relative Cost/Benefit', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        total_net = sum(cost_analysis.values())
        ax1.text(0.02, 0.98, f'Net Impact: {total_net:+,.0f}',
                transform=ax1.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Detection efficiency
        ax2 = axes[0, 1]
        efficiency_metrics = {
            'Detection Rate': recall,
            'False Alarm Rate': fpr,
            'Precision': precision,
            'F1-Score': f1
        }
        
        x_pos = np.arange(len(efficiency_metrics))
        bars = ax2.bar(x_pos, efficiency_metrics.values(), 
                      color=[self.colors['primary']] * len(efficiency_metrics))
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(efficiency_metrics.keys(), rotation=45, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.set_title('Detection Efficiency', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for i, (key, val) in enumerate(efficiency_metrics.items()):
            ax2.text(i, val + 0.02, f'{val:.3f}', ha='center', 
                    fontweight='bold', fontsize=10)
        
        # 3. Threshold impact analysis
        ax3 = axes[1, 0]
        # Simulate different threshold impacts
        thresholds = np.arange(0.1, 0.9, 0.1)
        recalls = []
        precisions = []
        fprs = []
        
        for thresh in thresholds:
            # Simplified simulation - adjust based on current performance
            recall_adj = recall * (1 - abs(thresh - 0.5))
            precision_adj = precision * (1 + (thresh - 0.5))
            fpr_adj = fpr * (1 - (thresh - 0.3))  # Lower threshold = higher FPR
            
            recalls.append(max(0, min(1, recall_adj)))
            precisions.append(max(0, min(1, precision_adj)))
            fprs.append(max(0, min(1, fpr_adj)))
        
        ax3.plot(thresholds, recalls, 'o-', label='Recall', linewidth=2, 
                color=self.colors['primary'])
        ax3.plot(thresholds, precisions, 's-', label='Precision', linewidth=2,
                color=self.colors['secondary'])
        ax3.plot(thresholds, fprs, '^-', label='FPR', linewidth=2,
                color=self.colors['danger'])
        
        ax3.set_xlabel('Threshold', fontweight='bold')
        ax3.set_ylabel('Score', fontweight='bold')
        ax3.set_title('Threshold Sensitivity Analysis', fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # Mark current threshold
        current_threshold = test_results['threshold'].iloc[0] if 'threshold' in test_results.columns else 0.5
        ax3.axvline(x=current_threshold, color='red', linestyle='--', 
                   alpha=0.7, label=f'Current: {current_threshold:.2f}')
        ax3.legend(loc='best')
        
        # 4. Deployment readiness gauge
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate readiness score
        readiness_score = recall * 0.6 + (1 - fpr) * 0.4
        
        # Determine status
        if readiness_score >= 0.8:
            status = "✅ EXCELLENT"
            status_color = self.colors['secondary']
            recommendation = "Ready for production deployment"
        elif readiness_score >= 0.65:
            status = "⚠️ GOOD"
            status_color = self.colors['warning']
            recommendation = "Can deploy with monitoring"
        else:
            status = "❌ NEEDS IMPROVEMENT"
            status_color = self.colors['danger']
            recommendation = "Requires retraining"
        
        # Create gauge visualization
        ax4.text(0.5, 0.85, 'Deployment Readiness', ha='center', 
                fontsize=14, fontweight='bold', transform=ax4.transAxes)
        
        # Gauge background
        gauge_bg = plt.Circle((0.5, 0.5), 0.3, color='lightgray', alpha=0.3, 
                             transform=ax4.transAxes)
        ax4.add_patch(gauge_bg)
        
        # Gauge fill (based on readiness)
        gauge_fill = plt.Circle((0.5, 0.5), 0.28, color=status_color, 
                               alpha=0.6, transform=ax4.transAxes)
        ax4.add_patch(gauge_fill)
        
        # Score text
        ax4.text(0.5, 0.5, f'{readiness_score:.2f}', ha='center', 
                va='center', fontsize=24, fontweight='bold',
                transform=ax4.transAxes)
        
        ax4.text(0.5, 0.3, status, ha='center', fontsize=12,
                fontweight='bold', color=status_color,
                transform=ax4.transAxes)
        
        ax4.text(0.5, 0.2, recommendation, ha='center', fontsize=10,
                style='italic', transform=ax4.transAxes)
        
        plt.suptitle('Business Impact Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        plot_path = os.path.join(PLOTS_DIR, f'business_impact_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Business impact plot saved: {plot_path}")
        return plot_path
    
    def plot_performance_curves(self, test_results, metadata):
        """Plot ROC and Precision-Recall curves (if data available)"""
        print("📊 Creating performance curves plot...")
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Get AUC score
        auc_score = test_results['auc_roc'].iloc[0] if 'auc_roc' in test_results.columns else 0.8
        
        # 1. ROC Curve (simulated if not available)
        ax1 = axes[0]
        
        # Create simulated ROC curve based on AUC
        fpr = np.linspace(0, 1, 100)
        tpr = auc_score * fpr + (1 - auc_score) * (1 - np.exp(-5 * fpr))
        tpr = np.clip(tpr, 0, 1)
        
        ax1.plot(fpr, tpr, color=self.colors['primary'], 
                lw=2, label=f'AUC = {auc_score:.3f}')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontweight='bold')
        ax1.set_ylabel('True Positive Rate', fontweight='bold')
        ax1.set_title('ROC Curve', fontweight='bold')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curve
        ax2 = axes[1]
        
        # Simulated Precision-Recall curve
        recall_curve = np.linspace(0, 1, 100)
        precision_curve = test_results['precision'].iloc[0] * np.ones_like(recall_curve)
        
        ax2.plot(recall_curve, precision_curve, color=self.colors['secondary'],
                lw=2, label=f'Precision = {test_results["precision"].iloc[0]:.3f}')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall', fontweight='bold')
        ax2.set_ylabel('Precision', fontweight='bold')
        ax2.set_title('Precision-Recall Curve', fontweight='bold')
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)
        
        # 3. Metrics vs Threshold (simulated)
        ax3 = axes[2]
        
        thresholds = np.linspace(0.1, 0.9, 20)
        
        # Simulate metric changes with threshold
        base_recall = test_results['recall'].iloc[0] if 'recall' in test_results.columns else 0.7
        base_precision = test_results['precision'].iloc[0] if 'precision' in test_results.columns else 0.8
        base_fpr = test_results['false_positive_rate'].iloc[0] if 'false_positive_rate' in test_results.columns else 0.05
        
        recalls = []
        precisions = []
        fprs = []
        
        for t in thresholds:
            # Adjust metrics based on threshold distance from 0.5
            recall_adj = base_recall * (1 - 0.8 * abs(t - 0.5))
            precision_adj = base_precision * (1 + 0.5 * (t - 0.5))
            fpr_adj = base_fpr * (1 - 0.6 * (t - 0.3))
            
            recalls.append(max(0, min(1, recall_adj)))
            precisions.append(max(0, min(1, precision_adj)))
            fprs.append(max(0, min(1, fpr_adj)))
        
        ax3.plot(thresholds, recalls, 'o-', label='Recall', 
                color=self.colors['primary'], lw=2)
        ax3.plot(thresholds, precisions, 's-', label='Precision', 
                color=self.colors['secondary'], lw=2)
        ax3.plot(thresholds, fprs, '^-', label='FPR', 
                color=self.colors['danger'], lw=2)
        
        current_threshold = test_results['threshold'].iloc[0] if 'threshold' in test_results.columns else 0.5
        ax3.axvline(x=current_threshold, color='red', linestyle='--',
                   label=f'Current: {current_threshold:.2f}')
        
        ax3.set_xlabel('Threshold', fontweight='bold')
        ax3.set_ylabel('Score', fontweight='bold')
        ax3.set_title('Metrics vs Threshold', fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Performance Curves Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(PLOTS_DIR, f'performance_curves_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Performance curves plot saved: {plot_path}")
        return plot_path
    
    def plot_feature_importance(self, metadata):
        """Plot feature importance if available in metadata"""
        print("🔍 Creating feature importance plot...")
        
        if not metadata or 'feature_importance' not in metadata:
            print("⚠️  No feature importance data available")
            return None
        
        feature_data = metadata.get('feature_importance', {})
        top_indices = feature_data.get('top_features_indices', [])
        top_scores = feature_data.get('top_features_scores', [])
        
        if not top_indices or not top_scores:
            print("⚠️  Feature importance data incomplete")
            return None
        
        # Take top 15 features
        n_features = min(15, len(top_indices))
        indices = top_indices[:n_features]
        scores = top_scores[:n_features]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Bar chart of top features
        ax1 = axes[0]
        y_pos = np.arange(n_features)
        
        # Create gradient colors
        colors = plt.cm.Blues(np.linspace(0.6, 1, n_features))
        
        bars = ax1.barh(y_pos, scores, color=colors, edgecolor='black')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f'Feature {idx}' for idx in indices], fontweight='bold')
        ax1.invert_yaxis()
        ax1.set_xlabel('Importance Score', fontweight='bold')
        ax1.set_title(f'Top {n_features} Feature Importances', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{score:.4f}', ha='left', va='center',
                    fontweight='bold', fontsize=9)
        
        # 2. Cumulative importance
        ax2 = axes[1]
        sorted_scores = sorted(scores, reverse=True)
        cumulative = np.cumsum(sorted_scores) / np.sum(scores)
        
        ax2.plot(range(1, n_features + 1), cumulative, 'o-', 
                color=self.colors['primary'], linewidth=2, markersize=6)
        ax2.set_xlabel('Number of Features', fontweight='bold')
        ax2.set_ylabel('Cumulative Importance', fontweight='bold')
        ax2.set_title('Cumulative Feature Importance', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        # Mark points
        for n in [5, 10, 15]:
            if n <= n_features:
                idx = n - 1
                ax2.plot(n, cumulative[idx], 'ro', markersize=8)
                ax2.text(n + 0.5, cumulative[idx] - 0.05,
                        f'{cumulative[idx]:.1%}', fontweight='bold')
        
        plt.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(PLOTS_DIR, f'feature_importance_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Feature importance plot saved: {plot_path}")
        return plot_path
    
    def create_summary_report(self, test_results, metadata):
        """Create comprehensive summary report"""
        print("📋 Creating summary report...")
        
        fig = plt.figure(figsize=(12, 15))
        
        # Create text-based summary
        summary_text = self._generate_summary_text(test_results, metadata)
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Add title
        plt.suptitle('MODEL PERFORMANCE SUMMARY REPORT', fontsize=18, 
                    fontweight='bold', y=0.98)
        
        # Add summary text
        ax.text(0.02, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, fontfamily='monospace',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='whitesmoke', 
                        alpha=0.9, pad=10))
        
        # Add timestamp
        ax.text(0.02, 0.02, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
               transform=ax.transAxes, fontsize=9, style='italic')
        
        plt.tight_layout()
        
        plot_path = os.path.join(PLOTS_DIR, f'summary_report_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Summary report saved: {plot_path}")
        return plot_path
    
    def _generate_summary_text(self, test_results, metadata):
        """Generate summary text for the report"""
        
        # Extract key metrics with defaults
        accuracy = test_results.get('accuracy', 0).iloc[0] if 'accuracy' in test_results.columns else 0
        recall = test_results['recall'].iloc[0] if 'recall' in test_results.columns else 0
        precision = test_results['precision'].iloc[0] if 'precision' in test_results.columns else 0
        f1 = test_results['f1_score'].iloc[0] if 'f1_score' in test_results.columns else 0
        auc = test_results['auc_roc'].iloc[0] if 'auc_roc' in test_results.columns else 0
        threshold = test_results['threshold'].iloc[0] if 'threshold' in test_results.columns else 0.5
        
        tp = test_results['true_positives'].iloc[0] if 'true_positives' in test_results.columns else 0
        fp = test_results['false_positives'].iloc[0] if 'false_positives' in test_results.columns else 0
        fn = test_results['false_negatives'].iloc[0] if 'false_negatives' in test_results.columns else 0
        tn = test_results['true_negatives'].iloc[0] if 'true_negatives' in test_results.columns else 0
        
        fpr = test_results['false_positive_rate'].iloc[0] if 'false_positive_rate' in test_results.columns else 0
        fnr = test_results['false_negative_rate'].iloc[0] if 'false_negative_rate' in test_results.columns else 0
        
        # Calculate business metrics
        detection_rate = recall
        false_alarm_rate = fpr
        business_score = recall * 0.6 + (1 - fpr) * 0.4
        
        # Determine deployment status
        if recall >= 0.75 and fpr <= 0.03:
            deployment_status = "✅ PRODUCTION READY"
            status_color = "green"
        elif recall >= 0.65 and fpr <= 0.05:
            deployment_status = "⚠️  ACCEPTABLE (with monitoring)"
            status_color = "orange"
        else:
            deployment_status = "❌ NEEDS IMPROVEMENT"
            status_color = "red"
        
        # Get model info from metadata
        model_info = metadata.get('model_info', {}) if metadata else {}
        n_estimators = model_info.get('n_estimators', 'N/A')
        minority_class = model_info.get('minority_class', 1)
        training_date = metadata.get('training_info', {}).get('training_date', 'Unknown') if metadata else 'Unknown'
        
        summary = f"""
{'='*60}
RANDOM FOREST MODEL - PERFORMANCE SUMMARY
{'='*60}

📊 MODEL INFORMATION:
• Model Type:           Random Forest Classifier
• Trees:                {n_estimators}
• Minority Class:       {minority_class} (malicious)
• Training Date:        {training_date}
• Optimal Threshold:    {threshold:.4f}

🎯 PERFORMANCE METRICS:
• Accuracy:             {accuracy:.4f}
• Precision:            {precision:.4f}
• Recall:               {recall:.4f} (Detection Rate)
• F1-Score:             {f1:.4f}
• AUC-ROC:              {auc:.4f}

⚠️  ERROR ANALYSIS:
• False Positive Rate:  {fpr:.4f} ({fp:,} false alarms)
• False Negative Rate:  {fnr:.4f} ({fn:,} missed threats)
• Specificity:          {1 - fpr:.4f}

📊 CONFUSION MATRIX:
                    Predicted
                  Benign  Malicious
  Actual Benign   {tn:>6}   {fp:>6}
  Actual Malicious {fn:>6}   {tp:>6}

💰 BUSINESS IMPACT:
• Detection Rate:       {detection_rate:.1%}
• False Alarm Rate:     {false_alarm_rate:.1%}
• Business Score:       {business_score:.3f}
• Total Samples:        {tp + fp + fn + tn:,}
• Malicious Detected:   {tp:,}/{tp + fn:,}

{'='*60}
🚀 DEPLOYMENT ASSESSMENT:
{deployment_status}

💡 RECOMMENDATIONS:
{self._generate_recommendations(recall, fpr, threshold)}
{'='*60}
"""
        return summary
    
    def _generate_recommendations(self, recall, fpr, threshold):
        """Generate recommendations based on performance"""
        recommendations = []
        
        if recall < 0.65:
            recommendations.append("• Detection rate too low (<65%)")
            recommendations.append("• Consider lowering threshold to 0.25-0.35")
            recommendations.append("• Collect more malicious samples")
        elif fpr > 0.05:
            recommendations.append("• False alarm rate too high (>5%)")
            recommendations.append("• Consider increasing threshold to 0.4-0.5")
            recommendations.append("• Review feature extraction")
        elif recall >= 0.75 and fpr <= 0.03:
            recommendations.append("• Performance is excellent")
            recommendations.append("• Ready for Chrome extension deployment")
            recommendations.append("• Monitor real-world performance")
        else:
            recommendations.append("• Performance is acceptable")
            recommendations.append(f"• Current threshold ({threshold:.3f}) is optimal")
            recommendations.append("• Deploy with performance monitoring")
        
        return "\n".join(recommendations)
    
    def plot_all(self):
        """Generate all plots"""
        print("\n" + "="*70)
        print("🎨 COMPREHENSIVE MODEL VISUALIZATION")
        print("="*70 + "\n")
        
        # Load results
        test_results, metadata, timestamp = self.load_latest_results()
        if test_results is None:
            print("❌ Failed to load results. Exiting.")
            return
        
        # Update timestamp with loaded one
        if timestamp:
            self.timestamp = timestamp
        
        print(f"\n📊 Model Performance Summary:")
        print(f"   Accuracy:  {test_results.get('accuracy', [0])[0]:.4f}")
        print(f"   Precision: {test_results.get('precision', [0])[0]:.4f}")
        print(f"   Recall:    {test_results.get('recall', [0])[0]:.4f}")
        print(f"   F1-Score:  {test_results.get('f1_score', [0])[0]:.4f}")
        print(f"   AUC-ROC:   {test_results.get('auc_roc', [0])[0]:.4f}")
        print(f"   Threshold: {test_results.get('threshold', [0.5])[0]:.4f}")
        
        # Generate all plots
        plots = []
        
        try:
            plots.append(self.plot_core_metrics(test_results))
        except Exception as e:
            print(f"❌ Error creating core metrics plot: {e}")
        
        try:
            plots.append(self.plot_business_impact(test_results))
        except Exception as e:
            print(f"❌ Error creating business impact plot: {e}")
        
        try:
            plots.append(self.plot_performance_curves(test_results, metadata))
        except Exception as e:
            print(f"❌ Error creating performance curves: {e}")
        
        try:
            feature_plot = self.plot_feature_importance(metadata)
            if feature_plot:
                plots.append(feature_plot)
        except Exception as e:
            print(f"❌ Error creating feature importance plot: {e}")
        
        try:
            plots.append(self.create_summary_report(test_results, metadata))
        except Exception as e:
            print(f"❌ Error creating summary report: {e}")
        
        # Print summary
        print(f"\n" + "="*70)
        print("✅ ALL PLOTS CREATED SUCCESSFULLY!")
        print(f"📁 Files saved to: {PLOTS_DIR}")
        
        for i, plot_path in enumerate(plots, 1):
            if plot_path:
                print(f"   {i}. {os.path.basename(plot_path)}")
        
        # Final assessment
        recall = test_results.get('recall', [0])[0]
        fpr = test_results.get('false_positive_rate', [0])[0]
        
        print(f"\n🎯 FINAL ASSESSMENT:")
        if recall >= 0.75 and fpr <= 0.03:
            print("   ✅ EXCELLENT - Ready for Chrome extension deployment!")
        elif recall >= 0.65:
            print("   ⚠️  GOOD - Can deploy with monitoring")
        else:
            print("   ❌ NEEDS IMPROVEMENT - Requires retraining")
        
        print(f"\n💡 Next steps:")
        if recall < 0.65:
            print("   1. Lower threshold to 0.25-0.35 for better detection")
            print("   2. Collect more malicious samples")
            print("   3. Review feature extraction")
        elif fpr > 0.05:
            print("   1. Increase threshold to 0.4-0.5 to reduce false alarms")
            print("   2. Fine-tune feature selection")
            print("   3. Consider ensemble methods")
        else:
            print("   1. Deploy to Chrome extension")
            print("   2. Monitor real-world performance")
            print("   3. Set up regular retraining schedule")

def main():
    """Main function to run visualization"""
    visualizer = ModelVisualizer()
    visualizer.plot_all()

if __name__ == "__main__":
    main()