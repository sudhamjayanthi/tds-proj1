@Dhruv Gupta said: I have doubt in question 10 to convert pdf to markdown  

I am not getting correct markdown  

[@pds\_staff](/u/pds_staff)


@Ashutosh Banerjee said: Try using the pymupdf4llm Library  

pip install pymupdf4llm


import pymupdf4llm  

md\_text \= pymupdf4llm.to\_markdown(“input.pdf”)


import pathlib  

pathlib.Path(“output.md”).write\_bytes(md\_text.encode())


import pymupdf4llm  

llama\_reader \= pymupdf4llm.LlamaMarkdownReader()  

llama\_docs \= llama\_reader.load\_data(“input.pdf”)

