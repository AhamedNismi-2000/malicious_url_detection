"""
Feature Extractor Module - Extracts heuristic features from URLs
Author: UWU/CST/21/083
"""

from .url_parser import URLParser
import re
from urllib.parse import unquote

class FeatureExtractor:
    def __init__(self):
        self.parser = URLParser()
        self.suspicious_keywords = [
            'login', 'signin', 'secure', 'verify', 'update', 'bank',
            'account', 'password', 'paypal', 'ebay', 'amazon', 'wallet',
            'confirm', 'validation', 'authenticate', 'security', 'admin'
        ]
        
        self.suspicious_tlds = ['.xyz', '.top', '.loan', '.tk', '.ml', 
                               '.ga', '.cf', '.gq', '.rest', '.download']
        
        self.hex_pattern = r'%[0-9a-fA-F]{2}'
        self.digit_pattern = r'\d'
    
    def extract_all_features(self, url):
        """Extract all heuristic features from URL"""
        features = {}
        
        # Structural features
        features['url_length'] = self.parser.get_url_length(url)
        features['domain_length'] = self.parser.get_domain_length(url)
        features['dot_count'] = self.parser.count_dots(url)
        features['hyphen_count'] = self.parser.count_hyphens(url)
        features['subdomain_count'] = self.parser.get_subdomain_count(url)
        features['path_depth'] = self.parser.get_path_depth(url)
        
        # Binary features
        features['has_ip'] = 1 if self.parser.has_ip_address(url) else 0
        features['has_special_chars'] = 1 if self.parser.has_special_chars(url) else 0
        features['has_redirect'] = 1 if self.parser.has_redirect(url) else 0
        features['has_https'] = 1 if url.startswith('https') else 0
        
        # Pattern features
        features['hex_encoded'] = self._has_hex_encoding(url)
        features['digit_ratio'] = self._get_digit_ratio(url)
        features['suspicious_tld'] = self._has_suspicious_tld(url)
        features['suspicious_keyword'] = self._has_suspicious_keyword(url)
        features['shortened_url'] = self._is_shortened_url(url)
        
        # Complexity features
        features['entropy'] = self.parser.get_entropy(url)
        features['avg_token_length'] = self._get_avg_token_length(url)
        
        # Additional lexical features
        parsed = self.parser.parse_url(url)
        features['domain_token_count'] = self._count_tokens(parsed['domain'])
        features['path_token_count'] = self._count_tokens(parsed['path'])
        
        return features
    
    def _has_hex_encoding(self, url):
        """Check if URL has hex encoding"""
        decoded = unquote(url)
        return 1 if decoded != url else 0
    
    def _get_digit_ratio(self, url):
        """Calculate ratio of digits in URL"""
        if not url:
            return 0
        digits = re.findall(self.digit_pattern, url)
        return len(digits) / len(url)
    
    def _has_suspicious_tld(self, url):
        """Check if URL has suspicious TLD"""
        parsed = self.parser.parse_url(url)
        tld = '.' + parsed['suffix']
        return 1 if tld in self.suspicious_tlds else 0
    
    def _has_suspicious_keyword(self, url):
        """Check if URL contains suspicious keywords"""
        url_lower = url.lower()
        for keyword in self.suspicious_keywords:
            if keyword in url_lower:
                return 1
        return 0
    
    def _is_shortened_url(self, url):
        """Check if URL is from shortening service"""
        shortened_domains = ['bit.ly', 'tinyurl', 'goo.gl', 'ow.ly', 
                            'is.gd', 'buff.ly', 'adf.ly', 't.co']
        parsed = self.parser.parse_url(url)
        domain = parsed['domain'] + '.' + parsed['suffix']
        
        for short_domain in shortened_domains:
            if short_domain in domain:
                return 1
        return 0
    
    def _count_tokens(self, text):
        """Count tokens separated by dots, hyphens, or underscores"""
        if not text:
            return 0
        tokens = re.split(r'[\.\-_]', text)
        return len([t for t in tokens if t])
    
    def _get_avg_token_length(self, url):
        """Calculate average token length in URL"""
        parsed = self.parser.parse_url(url)
        full_text = parsed['domain'] + parsed['suffix'] + parsed['path']
        
        tokens = re.split(r'[\.\-_/]', full_text)
        tokens = [t for t in tokens if t]
        
        if not tokens:
            return 0
            
        total_length = sum(len(t) for t in tokens)
        return total_length / len(tokens)
    
    def get_feature_names(self):
        """Return list of all feature names"""
        sample_features = self.extract_all_features("https://example.com")
        return list(sample_features.keys())

# Test feature extractor
if __name__ == "__main__":
    extractor = FeatureExtractor()
    
    test_url = "http://paypa1-login-secure.verify-site.com/account/update?id=123"
    features = extractor.extract_all_features(test_url)
    
    print("Extracted Features:")
    for feature, value in features.items():
        print(f"{feature}: {value}")
    
    print(f"\nTotal features extracted: {len(features)}")