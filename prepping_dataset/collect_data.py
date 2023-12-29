''' use https://manybooks.net for pdfs '''
import PyPDF2
import re


class collect_data():
    def __init__(self, txt_file, pdf_file, page_start, page_finish):
        self.txt_file = txt_file
        self.pdf_file = pdf_file
        self.page_start = page_start
        self.page_finish = page_finish
        self.appended_count = 0
        
    def load_pdf(self):
        try:
            self.reader = PyPDF2.PdfReader(self.pdf_file)
            print(f"PDF loaded successfully: {self.pdf_file}")
        except Exception as e:
            print(f"Failed to load PDF: {e}")
        
    def append_txt(self, text):
        try:
             text.encode('utf-8')
            with open(self.txt_file, "a", encoding="utf-8") as f:
                f.write(text + '\n')
        except UnicodeEncodeError:
            # if UnicodeEncodeError, handle surrogate characters
            clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore')
            with open(self.txt_file, "a", encoding="utf-8") as f:
                f.write(clean_text + '\n')
            self.appended_count += 1     
        except Exception as e:
            print(f"Failed to append text: {e}")
            
    def clean_to_output(self):
        for i in range(self.page_start, self.page_finish + 1):
            try:
                page = self.reader.pages[i].extract_text()
                if page:
                    clean_page = re.sub(r'[\t\n]+', ' ', page)
                    self.append_txt(clean_page)
                else:
                    print(f"No text found on page {i}")
            except IndexError:
                print(f"Page {i} is out of range.")
            except Exception as e:
                print(f"Error processing page {i}: {e}")

    def execute(self):
        self.load_pdf()
        if hasattr(self, 'reader'):
            self.clean_to_output()
            print(f"Total {self.appended_count} pages appended to {self.txt_file}")


collector = collect_data(pdf_file="D:\\coding\llms\\pdfs\\Beyond-the-Door.pdf",
                         txt_file="D:\\coding\\llms\\sci_storys\\Beyond-the-Door.txt",
                         page_start=4, 
                         page_finish=12)
collector.execute()