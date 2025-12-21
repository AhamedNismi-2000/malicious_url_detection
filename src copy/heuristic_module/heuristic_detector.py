"""
Main Heuristic Detector - Orchestrates the entire heuristic detection process
Author: UWU/CST/21/083
"""

from .url_parser import URLParser
from .feature_extractor import FeatureExtractor
from .rule_engine import RuleEngine
import json
import pandas as pd
from typing import Dict, List, Any

class HeuristicDetector:
    def __init__(self, config_path=None):
        self.parser = URLParser()
        self.extractor = FeatureExtractor()
        self.rule_engine = RuleEngine(config_path)
        
    def analyze_url(self, url: str) -> Dict[str, Any]:
        """
        Complete analysis of a single URL
        
        Returns:
            Dictionary with analysis results
        """
        try:
            # Step 1: Parse URL
            parsed_info = self.parser.parse_url(url)
            
            # Step 2: Extract features
            features = self.extractor.extract_all_features(url)
            
            # Step 3: Apply rules and get score
            score, reasons, risk_level = self.rule_engine.apply_rules(features)
            
            # Step 4: Generate explanation
            explanation = self.rule_engine.explain_decision(score, reasons, risk_level)
            
            # Prepare result
            result = {
                'url': url,
                'domain': f"{parsed_info['domain']}.{parsed_info['suffix']}",
                'is_malicious': risk_level in ['high', 'critical'],
                'analysis': {
                    'parsed_info': parsed_info,
                    'features': features,
                    'score_details': explanation
                }
            }
            
            return result
            
        except Exception as e:
            return {
                'url': url,
                'error': str(e),
                'is_malicious': False,
                'analysis': None
            }
    
    def analyze_batch(self, urls: List[str]) -> pd.DataFrame:
        """
        Analyze multiple URLs
        
        Returns:
            DataFrame with analysis results
        """
        results = []
        
        for url in urls:
            result = self.analyze_url(url)
            
            if result['analysis']:
                row = {
                    'url': result['url'],
                    'domain': result['domain'],
                    'is_malicious': result['is_malicious'],
                    'risk_score': result['analysis']['score_details']['risk_score'],
                    'risk_level': result['analysis']['score_details']['risk_level'],
                    'violation_count': result['analysis']['score_details']['total_violations'],
                    'recommendation': result['analysis']['score_details']['recommendation']
                }
                results.append(row)
        
        return pd.DataFrame(results)
    
    def get_detailed_report(self, url: str) -> str:
        """Generate detailed text report for a URL"""
        result = self.analyze_url(url)
        
        if result.get('error'):
            return f"Error analyzing {url}: {result['error']}"
        
        analysis = result['analysis']
        score_details = analysis['score_details']
        
        report = f"""
{'='*70}
HEURISTIC ANALYSIS REPORT
{'='*70}

URL: {result['url']}
Domain: {result['domain']}
Risk Level: {score_details['risk_level'].upper()}
Risk Score: {score_details['risk_score']}
Classification: {'MALICIOUS' if result['is_malicious'] else 'SAFE'}

{'-'*70}
DETAILED ANALYSIS
{'-'*70}

URL Components:
  • Scheme: {analysis['parsed_info']['scheme']}
  • Domain: {analysis['parsed_info']['domain']}
  • TLD: {analysis['parsed_info']['suffix']}
  • Subdomain: {analysis['parsed_info']['subdomain'] or 'None'}
  • Path: {analysis['parsed_info']['path']}
  • Query: {analysis['parsed_info']['query'] or 'None'}

Key Features:
  • Length: {analysis['features']['url_length']} chars
  • Dots: {analysis['features']['dot_count']}
  • Subdomains: {analysis['features']['subdomain_count']}
  • Has IP: {'Yes' if analysis['features']['has_ip'] else 'No'}
  • Has Special Chars: {'Yes' if analysis['features']['has_special_chars'] else 'No'}
  • Entropy: {analysis['features']['entropy']:.2f}

{'-'*70}
RULE VIOLATIONS ({score_details['total_violations']} found)
{'-'*70}
"""
        
        for i, violation in enumerate(score_details['rule_violations'], 1):
            report += f"{i}. {violation}\n"
        
        report += f"""
{'-'*70}
RECOMMENDATION
{'-'*70}
{score_details['recommendation']}

{'='*70}
"""
        
        return report
    
    def save_config(self, filepath: str):
        """Save current configuration to YAML file"""
        import yaml
        
        config = {
            'thresholds': self.rule_engine.config['thresholds'],
            'weights': self.rule_engine.config['weights'],
            'risk_levels': self.rule_engine.config['risk_levels']
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Configuration saved to {filepath}")
    
    def export_features(self, urls: List[str], output_file: str):
        """Export extracted features to CSV for analysis"""
        data = []
        
        for url in urls:
            features = self.extractor.extract_all_features(url)
            features['url'] = url
            data.append(features)
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"Features exported to {output_file}")
        
        return df

# Main execution
if __name__ == "__main__":
    # Initialize detector
    detector = HeuristicDetector()
    
    # Test URLs
    test_urls = [
        "https://www.google.com",
        "http://192.168.1.1:8080/login",
        "https://paypa1-verify.login-secure.xyz/account/update?redirect=malicious.com",
        "https://github.com/microsoft/vscode",
        "http://free-gift-card.reward.site.top/get-gift"
    ]
    
    print("Testing Heuristic Detector...\n")
    
    # Analyze each URL
    for url in test_urls:
        report = detector.get_detailed_report(url)
        print(report)
    
    # Batch analysis
    print("\n\nBatch Analysis Results:")
    df = detector.analyze_batch(test_urls)
    print(df[['url', 'risk_level', 'risk_score', 'is_malicious']])
    
    # Save configuration example
    detector.save_config("heuristic_config.yaml")