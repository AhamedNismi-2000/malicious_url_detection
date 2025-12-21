"""
Rule Engine - Applies heuristic rules and scores URLs
Author: UWU/CST/21/083
"""

import yaml
from typing import Dict, List, Tuple

class RuleEngine:
    def __init__(self, config_path=None):
        # Default configuration
        self.config = {
            'thresholds': {
                'url_length': {'warning': 75, 'danger': 100},
                'dot_count': {'warning': 4, 'danger': 6},
                'subdomain_count': {'warning': 3, 'danger': 5},
                'hyphen_count': {'warning': 2, 'danger': 4},
                'path_depth': {'warning': 3, 'danger': 5},
                'digit_ratio': {'warning': 0.3, 'danger': 0.5},
                'entropy': {'warning': 3.5, 'danger': 4.5}
            },
            'weights': {
                'has_ip': 2.0,
                'has_special_chars': 1.5,
                'has_redirect': 1.8,
                'suspicious_tld': 1.7,
                'suspicious_keyword': 1.6,
                'shortened_url': 1.4,
                'hex_encoded': 1.9,
                'url_length': 0.1,  # per character over warning
                'dot_count': 0.3,   # per dot over warning
                'subdomain_count': 0.4,
                'hyphen_count': 0.3,
                'high_digit_ratio': 1.2,
                'high_entropy': 0.2  # per 0.1 over warning
            },
            'risk_levels': {
                'low': 2.0,
                'medium': 4.0,
                'high': 6.0,
                'critical': 8.0
            }
        }
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
            
        # Update config with loaded values
        for key in loaded_config:
            if key in self.config:
                if isinstance(self.config[key], dict):
                    self.config[key].update(loaded_config[key])
                else:
                    self.config[key] = loaded_config[key]
    
    def apply_rules(self, features: Dict) -> Tuple[float, List[str], str]:
        """
        Apply heuristic rules to features and return score
        
        Returns:
            score: float - Total risk score
            reasons: List[str] - List of rule violations
            risk_level: str - 'low', 'medium', 'high', or 'critical'
        """
        score = 0.0
        reasons = []
        
        # Rule 1: URL Length
        url_len = features.get('url_length', 0)
        if url_len > self.config['thresholds']['url_length']['danger']:
            score += self.config['weights']['url_length'] * 3
            reasons.append(f"Very long URL ({url_len} chars)")
        elif url_len > self.config['thresholds']['url_length']['warning']:
            score += self.config['weights']['url_length'] * 2
            reasons.append(f"Long URL ({url_len} chars)")
        
        # Rule 2: IP Address
        if features.get('has_ip', 0) == 1:
            score += self.config['weights']['has_ip']
            reasons.append("Contains IP address")
        
        # Rule 3: Special Characters
        if features.get('has_special_chars', 0) == 1:
            score += self.config['weights']['has_special_chars']
            reasons.append("Contains special characters in domain")
        
        # Rule 4: Suspicious Keywords
        if features.get('suspicious_keyword', 0) == 1:
            score += self.config['weights']['suspicious_keyword']
            reasons.append("Contains suspicious keywords")
        
        # Rule 5: Dot Count
        dot_count = features.get('dot_count', 0)
        if dot_count > self.config['thresholds']['dot_count']['danger']:
            score += self.config['weights']['dot_count'] * 3
            reasons.append(f"Too many dots ({dot_count})")
        elif dot_count > self.config['thresholds']['dot_count']['warning']:
            score += self.config['weights']['dot_count'] * 2
            reasons.append(f"Many dots ({dot_count})")
        
        # Rule 6: Subdomain Count
        subdomain_count = features.get('subdomain_count', 0)
        if subdomain_count > self.config['thresholds']['subdomain_count']['danger']:
            score += self.config['weights']['subdomain_count'] * 3
            reasons.append(f"Too many subdomains ({subdomain_count})")
        elif subdomain_count > self.config['thresholds']['subdomain_count']['warning']:
            score += self.config['weights']['subdomain_count'] * 2
            reasons.append(f"Many subdomains ({subdomain_count})")
        
        # Rule 7: Suspicious TLD
        if features.get('suspicious_tld', 0) == 1:
            score += self.config['weights']['suspicious_tld']
            reasons.append("Uses suspicious top-level domain")
        
        # Rule 8: Hex Encoding
        if features.get('hex_encoded', 0) == 1:
            score += self.config['weights']['hex_encoded']
            reasons.append("Contains hex encoding")
        
        # Rule 9: Redirect
        if features.get('has_redirect', 0) == 1:
            score += self.config['weights']['has_redirect']
            reasons.append("Contains redirect parameters")
        
        # Rule 10: Shortened URL
        if features.get('shortened_url', 0) == 1:
            score += self.config['weights']['shortened_url']
            reasons.append("Uses URL shortening service")
        
        # Rule 11: Digit Ratio
        digit_ratio = features.get('digit_ratio', 0)
        if digit_ratio > self.config['thresholds']['digit_ratio']['danger']:
            score += self.config['weights']['high_digit_ratio'] * 2
            reasons.append(f"High digit ratio ({digit_ratio:.2f})")
        elif digit_ratio > self.config['thresholds']['digit_ratio']['warning']:
            score += self.config['weights']['high_digit_ratio']
            reasons.append(f"Moderate digit ratio ({digit_ratio:.2f})")
        
        # Rule 12: Entropy (complexity)
        entropy = features.get('entropy', 0)
        if entropy > self.config['thresholds']['entropy']['danger']:
            extra = entropy - self.config['thresholds']['entropy']['danger']
            score += self.config['weights']['high_entropy'] * (extra * 10)
            reasons.append(f"High entropy/complexity ({entropy:.2f})")
        elif entropy > self.config['thresholds']['entropy']['warning']:
            extra = entropy - self.config['thresholds']['entropy']['warning']
            score += self.config['weights']['high_entropy'] * (extra * 10)
            reasons.append(f"Moderate entropy/complexity ({entropy:.2f})")
        
        # Rule 13: Hyphen Count
        hyphen_count = features.get('hyphen_count', 0)
        if hyphen_count > self.config['thresholds']['hyphen_count']['danger']:
            score += self.config['weights']['hyphen_count'] * 3
            reasons.append(f"Too many hyphens ({hyphen_count})")
        elif hyphen_count > self.config['thresholds']['hyphen_count']['warning']:
            score += self.config['weights']['hyphen_count'] * 2
            reasons.append(f"Many hyphens ({hyphen_count})")
        
        # Rule 14: Path Depth
        path_depth = features.get('path_depth', 0)
        if path_depth > self.config['thresholds']['path_depth']['danger']:
            reasons.append(f"Deep path structure ({path_depth} levels)")
            score += 1.5
        
        # Determine risk level
        risk_level = self._get_risk_level(score)
        
        return score, reasons, risk_level
    
    def _get_risk_level(self, score: float) -> str:
        """Convert score to risk level"""
        if score >= self.config['risk_levels']['critical']:
            return 'critical'
        elif score >= self.config['risk_levels']['high']:
            return 'high'
        elif score >= self.config['risk_levels']['medium']:
            return 'medium'
        else:
            return 'low'
    
    def explain_decision(self, score: float, reasons: List[str], risk_level: str) -> Dict:
        """Generate detailed explanation of the decision"""
        explanation = {
            'risk_score': round(score, 2),
            'risk_level': risk_level,
            'rule_violations': reasons,
            'total_violations': len(reasons),
            'recommendation': self._get_recommendation(risk_level)
        }
        
        return explanation
    
    def _get_recommendation(self, risk_level: str) -> str:
        """Get recommendation based on risk level"""
        recommendations = {
            'low': 'This URL appears safe. Proceed with normal caution.',
            'medium': 'This URL shows some suspicious characteristics. Exercise caution.',
            'high': 'This URL is highly suspicious. Avoid entering personal information.',
            'critical': 'DANGER - This URL shows multiple malicious characteristics. DO NOT PROCEED.'
        }
        return recommendations.get(risk_level, 'Unknown risk level')

# Test the rule engine
if __name__ == "__main__":
    from feature_extractor import FeatureExtractor
    
    # Create test
    extractor = FeatureExtractor()
    rule_engine = RuleEngine()
    
    test_urls = [
        "https://www.google.com/search?q=hello",
        "http://192.168.1.1/login.php",
        "https://paypa1-login.secure-verify.site.xyz/account/update?id=%20redirect",
        "http://bit.ly/3xyz123",
        "https://my-bank-update.secure-login.site.com/verify/account/password/reset"
    ]
    
    for url in test_urls:
        print(f"\n{'='*60}")
        print(f"Analyzing: {url}")
        print('='*60)
        
        features = extractor.extract_all_features(url)
        score, reasons, risk_level = rule_engine.apply_rules(features)
        explanation = rule_engine.explain_decision(score, reasons, risk_level)
        
        print(f"Risk Score: {explanation['risk_score']}")
        print(f"Risk Level: {explanation['risk_level'].upper()}")
        print(f"Violations Found: {explanation['total_violations']}")
        print("\nReasons:")
        for reason in explanation['rule_violations']:
            print(f"  • {reason}")
        print(f"\nRecommendation: {explanation['recommendation']}")