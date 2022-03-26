=== "Memaid"
	``` mermaid
	graph LR
	  A[Start] --> B{Error?};
	  B -->|Yes| C[Hmm...];
	  C --> D[Debug];
	  D --> B;
	  B ---->|No| E[Yay!];
	```
=== "source"
	```
	graph LR
	  A[Start] --> B{Error?};
	  B -->|Yes| C[Hmm...];
	  C --> D[Debug];
	  D --> B;
	  B ---->|No| E[Yay!];
	```
	
===! "source"
	test
	```
	code
	```
	

- [x] demo