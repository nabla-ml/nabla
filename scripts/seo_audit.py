#!/usr/bin/env python3
"""
SEO Audit Script for Nabla Documentation
Checks various SEO aspects and provides recommendations.
"""

import requests
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse
import json
import sys
from pathlib import Path
import time

class SEOAuditor:
    def __init__(self, base_url="https://nablaml.com"):
        self.base_url = base_url.rstrip('/')
        self.issues = []
        self.warnings = []
        self.successes = []
        
    def log_issue(self, issue):
        self.issues.append(issue)
        print(f"❌ ISSUE: {issue}")
        
    def log_warning(self, warning):
        self.warnings.append(warning)
        print(f"⚠️  WARNING: {warning}")
        
    def log_success(self, success):
        self.successes.append(success)
        print(f"✅ SUCCESS: {success}")
    
    def check_robots_txt(self):
        """Check robots.txt configuration"""
        print("\n🤖 Checking robots.txt...")
        try:
            response = requests.get(f"{self.base_url}/robots.txt", timeout=10)
            if response.status_code == 200:
                content = response.text
                if "Sitemap:" in content:
                    self.log_success("Robots.txt contains sitemap reference")
                else:
                    self.log_issue("Robots.txt missing sitemap reference")
                    
                if "Disallow: /_static/" in content:
                    self.log_success("Properly blocks static assets")
                    
                if "Allow: /api/" in content:
                    self.log_success("Explicitly allows API documentation")
                    
            else:
                self.log_issue(f"Robots.txt not accessible (Status: {response.status_code})")
        except Exception as e:
            self.log_issue(f"Failed to fetch robots.txt: {e}")
    
    def check_sitemap(self):
        """Check sitemap.xml and extract URLs"""
        print("\n🗺️  Checking sitemap.xml...")
        try:
            response = requests.get(f"{self.base_url}/sitemap.xml", timeout=10)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                urls = []
                
                # Parse sitemap URLs
                for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                    loc = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                    priority = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}priority')
                    lastmod = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod')
                    
                    url_info = {
                        'url': loc.text if loc is not None else '',
                        'priority': priority.text if priority is not None else '0.5',
                        'lastmod': lastmod.text if lastmod is not None else 'Unknown'
                    }
                    urls.append(url_info)
                
                self.log_success(f"Sitemap found with {len(urls)} URLs")
                
                # Check for important sections
                api_urls = [u for u in urls if '/api/' in u['url']]
                tutorial_urls = [u for u in urls if '/tutorials/' in u['url']]
                
                self.log_success(f"Found {len(api_urls)} API documentation URLs")
                self.log_success(f"Found {len(tutorial_urls)} tutorial URLs")
                
                # Check homepage priority
                homepage = next((u for u in urls if u['url'] == f"{self.base_url}/"), None)
                if homepage and float(homepage['priority']) == 1.0:
                    self.log_success("Homepage has maximum priority (1.0)")
                else:
                    self.log_warning("Homepage should have priority 1.0")
                
                return urls
            else:
                self.log_issue(f"Sitemap not accessible (Status: {response.status_code})")
                return []
        except Exception as e:
            self.log_issue(f"Failed to parse sitemap: {e}")
            return []
    
    def check_page_seo(self, url, max_retries=3):
        """Check SEO elements for a specific page"""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=15, headers={
                    'User-Agent': 'SEO Auditor Bot 1.0'
                })
                
                if response.status_code != 200:
                    if attempt == max_retries - 1:
                        self.log_warning(f"Page {url} returned status {response.status_code}")
                    continue
                
                html = response.text.lower()
                issues_found = []
                
                # Check critical SEO elements
                if '<title>' not in html:
                    issues_found.append("Missing title tag")
                elif 'nabla' not in html[html.find('<title>'):html.find('</title>')]:
                    issues_found.append("Title doesn't contain brand name")
                
                if 'name="description"' not in html:
                    issues_found.append("Missing meta description")
                
                if 'rel="canonical"' not in html:
                    issues_found.append("Missing canonical URL")
                
                if 'property="og:' not in html:
                    issues_found.append("Missing Open Graph tags")
                
                if 'name="twitter:' not in html:
                    issues_found.append("Missing Twitter Card tags")
                
                if 'application/ld+json' not in html:
                    issues_found.append("Missing structured data")
                
                # Check for duplicate meta viewport (known issue)
                viewport_count = html.count('name="viewport"')
                if viewport_count > 1:
                    issues_found.append(f"Duplicate viewport meta tags ({viewport_count} found)")
                
                return {
                    'url': url,
                    'status': response.status_code,
                    'issues': issues_found,
                    'title': self.extract_title(response.text),
                    'description': self.extract_meta_description(response.text)
                }
                
            except Exception as e:
                if attempt == max_retries - 1:
                    self.log_warning(f"Failed to check {url}: {e}")
                else:
                    time.sleep(1)  # Brief delay before retry
        
        return None
    
    def extract_title(self, html):
        """Extract page title"""
        import re
        match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else "No title found"
    
    def extract_meta_description(self, html):
        """Extract meta description"""
        import re
        match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']', html, re.IGNORECASE)
        return match.group(1).strip() if match else "No description found"
    
    def run_full_audit(self, sample_pages=10):
        """Run complete SEO audit"""
        print("🔍 Starting comprehensive SEO audit...\n")
        
        # Check basic infrastructure
        self.check_robots_txt()
        urls = self.check_sitemap()
        
        if not urls:
            print("❌ Cannot continue without sitemap data")
            return
        
        # Check sample of pages for detailed SEO
        print(f"\n📄 Checking SEO for sample of {min(sample_pages, len(urls))} pages...")
        
        # Always check homepage
        homepage_result = self.check_page_seo(self.base_url)
        if homepage_result:
            print(f"\n🏠 Homepage ({self.base_url}):")
            print(f"   Title: {homepage_result['title']}")
            print(f"   Description: {homepage_result['description'][:100]}...")
            if homepage_result['issues']:
                for issue in homepage_result['issues']:
                    self.log_issue(f"Homepage: {issue}")
        
        # Check important sections
        important_urls = []
        for pattern in ['/api/index.html', '/tutorials/index.html', '/api/array.html']:
            matching_urls = [u for u in urls if pattern in u['url']]
            important_urls.extend(matching_urls[:2])  # Max 2 per pattern
        
        # Add some random pages
        import random
        random.shuffle(urls)
        sample_urls = important_urls + urls[:sample_pages-len(important_urls)]
        
        page_results = []
        for url_info in sample_urls[:sample_pages]:
            if url_info['url'] == self.base_url:
                continue  # Already checked
            
            result = self.check_page_seo(url_info['url'])
            if result:
                page_results.append(result)
                if result['issues']:
                    print(f"\n📄 {result['url']}:")
                    for issue in result['issues']:
                        self.log_issue(f"Page {urlparse(result['url']).path}: {issue}")
        
        # Summary report
        self.print_summary_report(len(urls), page_results)
    
    def print_summary_report(self, total_urls, checked_pages):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("📊 SEO AUDIT SUMMARY")
        print("="*80)
        
        print(f"✅ Successes: {len(self.successes)}")
        for success in self.successes:
            print(f"   • {success}")
        
        print(f"\n⚠️  Warnings: {len(self.warnings)}")
        for warning in self.warnings:
            print(f"   • {warning}")
        
        print(f"\n❌ Issues: {len(self.issues)}")
        for issue in self.issues:
            print(f"   • {issue}")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        print("1. Fix duplicate viewport meta tags in Sphinx template")
        print("2. Implement dynamic meta descriptions per page")
        print("3. Ensure Twitter Card images are properly configured")
        print("4. Consider adding breadcrumb structured data")
        print("5. Monitor Core Web Vitals with Google PageSpeed Insights")
        
        # Overall health score
        total_checks = len(self.successes) + len(self.warnings) + len(self.issues)
        if total_checks > 0:
            health_score = (len(self.successes) / total_checks) * 100
            print(f"\n🏥 Overall SEO Health Score: {health_score:.1f}%")
            
            if health_score >= 80:
                print("🟢 Excellent SEO setup!")
            elif health_score >= 60:
                print("🟡 Good SEO setup with room for improvement")
            else:
                print("🔴 SEO needs significant attention")

if __name__ == "__main__":
    auditor = SEOAuditor()
    auditor.run_full_audit(sample_pages=15)
