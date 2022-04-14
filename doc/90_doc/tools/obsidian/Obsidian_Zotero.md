```ad-mypurple
title: 

[[Obsidian]]
[[Zotero]]


```

# 方式一：ob citations
通过citations 读取bib文件

[Zotero 和 obsidian 联动，无需 mdnotes、无需导出 markdown 实现原子化笔记、思维导图、pdf 跳转 - 知乎](https://zhuanlan.zhihu.com/p/439177612)

```ad-mypurple
title: 

zotero
- better Bibtex 
	- 设定citekey `[year]_[auth:lower]_[shorttitle2_2]`
	- ***bibtex导出字段要设定***
		- `abstract,note,month,doi,langid,keywords,timestamp,file`
- 设置Citations插件
	-   模板
	```
		---
		
		---
		```ad-info
		title: 文献信息
		
		- Title: =={{title}}==
		- Authors: {{authorString}}
		- Year: {{year}}
		- Journal: {{containerTitle}}
		- DOI: {{DOI}}
		- Tags: {{entry.data.fields.keywords}}
		- [Zotero Library]({{zoteroSelectURI}})
		```
		
		```ad-note
		title:Abstract
		
		{{abstract}}
		```
		
		file_path:
		=={{entry.data.fields.file}}==
		---
	```

```

# 方式二：mdnotes
- mdnote 模板
	- [Default Templates | Zotero-mdnotes](https://argentinaos.com/zotero-mdnotes/docs/advanced/templates/defaults/)
	- 多个文件导出
		- ==metadata条目- Zotero Metadata Template.md==
		- single note- Zotero Note Template.md
		- notes-Mdnotes Default Template.md
	- 将pdf和笔记合并为单个文件导出
		- default + note
		- Mdnotes Default Template.md
	- 修改模板
		- E:\000_Corner\.templates\mdnotes

		```metadata 
		---
		annotation-target: file://
		tags: #note, #library
		---
		
		## 题录
		
		* Title:: {{title}}
		* Journal:: {{publicationTitle}}
		* Authors:: {{author}}
		{{date}}
		{{dateAdded}}
		* Tags:: {{tags}}
		* DOI: {{DOI}}
		* Collections:: {{collections}}
		* LocalLibrary:{{localLibrary}}
		* CloudLibrary:{{cloudLibrary}}
		* PDF Attachments
			- {{pdfAttachments}}
		* Abstract:
			- {{abstractNote}}
		
		## Notes
		```
		- 修改mdnotes设置 placeholders
			- zotero 设置编辑器
			- [Docs](https://f8lfn9zs2l.feishu.cn/docs/doccnNsx8SjHN02Y14EsEOyUHKh)

	```json
	title
		{"content":"# {{field_contents}}", "field_contents": "{{content}}", "link_style": "no-links"}
		{"content":"{{field_contents}}", "field_contents": "{{content}}", "link_style": "no-links"}
		
	author
		{"content":"{{bullet}} Authors: {{field_contents}}", "link_style": "wiki", "list_separator": ", "}
		{"content":"{{field_contents}}", "link_style": "wiki", "list_separator": ", "}
		
	doi
		{"content":"{{bullet}} DOI: {{field_contents}}", "field_contents": "{{content}}", "link_style": "no-links"}
		{"content":"{{field_contents}}", "field_contents": "{{content}}", "link_style": "no-links"}
		
	Tags
		{"content":"{{bullet}} Tags: {{field_contents}}", "field_contents": "#{{content}}", "link_style": "no-links", "list_separator": ", ", "remove_spaces": "true"}
		{"content":"{{field_contents}}", "field_contents": "#{{content}}", "link_style": "no-links", "list_separator": ", ", "remove_spaces": "true"}
		
	collections
		{"content":"{{bullet}} Topics: {{field_contents}}", "field_contents": "{{content}}", "link_style": "wiki", "list_separator": ", "}
		{"content":"{{field_contents}}", "field_contents": "{{content}}", "link_style": "wiki", "list_separator": ", "}
		
	Related 
		{"content":"{{bullet}} Related: {{field_contents}}", "link_style": "wiki", "list_separator": ", "}
		{"content":"{{field_contents}}", "link_style": "wiki", "list_separator": ", "}
	
	LocalLibrary/CloudLibrary
		{"content":"{{bullet}} {{field_contents}}"}
		{"content":"{{field_contents}}","field_contents": "{{content}}"}
	
	abstractNote
		{"content":"## Abstract\n\n{{field_contents}}\n", "field_contents": "{{content}}", "link_style": "no-links", "list_separator": ", "}
		{"content":"{{field_contents}}", "field_contents": "{{content}}", "link_style": "no-links", "list_separator": ", "}
	
	pdfAttachments
		{"content":"{{bullet}} PDF Attachments\n\t- {{field_contents}}", "field_contents": "{{content}}", "list_separator": "\n\t- "}
		{"content":"{{field_contents}}", "field_contents": "{{content}}", "list_separator": "\n\t- "}
	
	notes
		{"content":"## Highlights and Annotations\n\n- {{field_contents}}", "field_contents": "{{content}}", "link_style": "wiki", "list_separator": "\n- "}
		{"content":"{{field_contents}}", "field_contents": "{{content}}", "link_style": "wiki", "list_separator": "\n- "}
		
	新建publicationTitle
		{"content":"{{field_contents}}", "field_contents": "{{content}}", "link_style": "wiki", "list_separator": ", "}
	```

- 构建查询统计语句
	- JS

	```dataviewjs
	let quotesPromises = dv.pages("#论文").map(async (p) => {  
	    const link = p.file.link;  
	    const title_array = [  
	        "#### 理论",  
	        "#### 内容",  
	        "#### 方法",  
	        "#### 结论",  
	        "#### 亮点",  
	        "#### 灵感",  
	        "***",  
	    ];  
	    var quotes_array = [];  
	    return await dv.io.load(p.file.link).then((contents) => {  
	        for (let index = 0; index < title_array.length - 1; index++) {  
	            const element_1 = title_array[index];  
	            const element_2 = title_array[index + 1];  
	  
	            const indexOfQuotes = contents.indexOf(element_1);  
	  
	            if (indexOfQuotes > 0) {  
	                const quoteStart = contents.substr(  
	                    indexOfQuotes + title_array[0].length  
	                );  
	                const quotes = quoteStart  
	                    .substr(0, quoteStart.indexOf(element_2))  
	                    .trim();  
	                quotes_array.push(quotes);  
	            } else {  
	                break;  
	            }  
	        }  
	        if (quotes_array.join === "") {  
	            return [];  
	        } else {  
				//console.log('------');  
				//console.log(quotes_array);  
	            quotes_array.unshift(p.file.link);  
				//console.log(quotes_array);  
				return Array(quotes_array);  
	        }  
	    });  
	});  
	let quotes = (await Promise.all(quotesPromises))  
	    .flat()  
	    .filter((q) => q.length > 0);  
	  
	dv.table(["文件", "理论", "内容", "方法", "结论", "亮点", "灵感"], quotes);  
	```
	- SQL
	```dataview
	TABLE Authors,Keywords,Journal,Date
	FROM "B900_Notes"
	SORT file.cday
	```
