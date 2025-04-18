import markdown

def load_rfp_review_wordlist():
    wordlist = []
    try:
        with open('config/review_wordlist.txt', 'r') as w:
            word_text = w.readlines()
            for text in word_text:
                target_word_part = text.split("[")[0].strip().strip('"')
                target_words = []
                if '/' in target_word_part:
                    target_words = [''.join(e for e in item.strip() if e.isalnum() or e.isspace()) for item in target_word_part.split('/')]
                else:
                    target_words = [target_word_part.strip()]
                    
                alternate_wordpart = text.split("versus")[1].strip()
                # alternatives = []
                alternate_word_items = alternate_wordpart.split('"')
                alternate_words = [item.strip() for item in alternate_word_items if item.strip() not in [',', 'or', '']]
                
                other_words = ", ".join(alternate_words)
                for target_word in target_words:
                    wordlist.append(target_word+": ["+ other_words +"]")

    except Exception as ex:
        print(ex)

    words = "\n".join(wordlist)
    return words

def convert_markdown_to_html(text):
    html = markdown.markdown(text)
    return html