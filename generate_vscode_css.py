import os
from pygments.formatters import HtmlFormatter

def build_css():
    style = HtmlFormatter(style='vsc').get_style_defs('.highlight')
    lines = []
    
    # Prefix every line with html[data-theme="dark"] to beat sphinx-book-theme specificity
    # Also add !important to all generated styles
    for line in style.split('\n'):
        if '{' in line:
            head, tail = line.split('{', 1)
            tail = tail.rsplit('}', 1)[0].strip()
            
            # Add !important to properties
            new_tail_parts = []
            for prop in tail.split(';'):
                prop = prop.strip()
                if prop:
                    new_tail_parts.append(prop + ' !important')
            
            # Combine
            new_head = f'html[data-theme="dark"] {head.strip()}'
            new_tail = '; '.join(new_tail_parts)
            if new_tail:
                new_tail += ';'
                
            lines.append(f'{new_head} {{ {new_tail} }}')
        else:
            lines.append(line)
            
    with open('docs/_static/vscode_dark.css', 'w') as f:
        f.write('\n'.join(lines))
        
if __name__ == '__main__':
    build_css()
