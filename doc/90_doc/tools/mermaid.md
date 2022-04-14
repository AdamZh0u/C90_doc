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
	
	

[Online FlowChart & Diagrams Editor - Mermaid Live Editor](https://mermaid-js.github.io/mermaid-live-editor)

# Flowchart

```ad-gray
- **nodes**
- the geometric shapes 
- **edges**
- the arrows or lines.

```

```
flowchart LR  
	 id[This is the text in the box]  
	 没有空格
	 可以放在引号内
```


```mermaid
flowchart LR
	 id1[text]-->id2([demo])
	 id1-->id3>demo]
	 id2-->id4{{xy}}
	 id3-->id5{123}
	 id5-->id6{{demo}}
```

## Node
- 方向
	- TD/TB
	- BT
	- LR
	- RL
```ad-mypurple
title: shape

- `[]`
- `()`
- `([])` stadium-shaped
- `[[]]`
- `[()]` cylindrical / datebase 
- `(())`
- `>]`
- `{}` rhombus
- `{{}}` hexagon
- `[//]` Parallelogram
- `[\\]` Parallelogram
- `[/\]`
- `[\/]`
```

## links
```ad-mypurple
title: links

- `-->` and `-->|text|` and `-- text -->`
- `---` and `---|text|` and `--text---`  
- dotted 
	```
	-.->
	-.text.-> 
	```
- thick
	```
	==>
	==text==>
	```
- multiple nodes links
	```
	a-->b&c-->d
	A&B-->C&D
	```
	
- new types
	```
	--o
	--x
	<-->
	x--x
	```
- `---->` 长度代表rank

```

## subgraphs
- basic 
```mermaid
flowchart TB  
	c1-->a2
	
	subgraph id1 [one]
		a1-->a2  
	end  
	
	subgraph two
		b1-->b2  
	end  
	
	subgraph three
	c1-->c2  
	end
```

- set edges to and from subgraphs
```mermaid
flowchart TB
    c1-->a2
	
    subgraph one
	    a1-->a2
    end
    
	subgraph two
	    b1-->b2
    end
    subgraph three
	    c1-->c2
    end
    
	one --> two
    three --> two
    two --> c2
```

- directions in subgraph
```mermaid
flowchart LR
	subgraph TOP
		direction TB
	
		subgraph B1
			direction RL
			i1 -->f1
		end
	
		subgraph B2
			direction BT
			i2 -->f2
		end
	end
	
	A --> TOP --> B
	B1 --> B2
```

## styling
```ad-mypurple
title: styling links

- linkstyle index ...
- stroke
	- `#ff3`
	-  



```

```mermaid
flowchart LR
	A-->B
	B-->C
	C-->D
	linkStyle 0 stroke:#ff3,stroke-width:4px,color:red;
	linkStyle 1 stroke:#ff3,stroke-width:4px,color:red;
	linkStyle 2 stroke:#FFA500,stroke-width:4px,color:red;
```

```ad-mypurple
title: styling nodes

- style id ...
- fill
- stroke
- stroke-width
- stroke-dasharray
```

```mermaid

flowchart LR
    id1(Start)-->id2(Stop)
    style id1 fill:#f9f,stroke:#333,stroke-width:4px
    style id2 fill:#bbf,stroke:#f66,stroke-width:2px,color:#fff,stroke-dasharray: 5 5
```

```ad-mypurple
title: classes

-  define a class of styles and attach this class to the nodes
	- `classDef className fill:#f9f,stroke:#333,stroke-width:4px;`
-  attach node to class
	- `class nodeId1,nodeId2 className;`
- shorter form `:::`
	- `A:::someclass --> B`
- CSS classes
- default class
	- `classDef default fill:#f9f,stroke:#333,stroke-width:4px;`
```

```mermaid
flowchart LR  
	A:::someclass --> B  
	B-->C
	C---D
	classDef default fill:#f9f,stroke:#333,stroke-width:2px;
	classDef someclass fill:#f96;
	classDef otherclass fill:#f9f,stroke:#333,stroke-width:4px;
	class C,D otherclass
```

## Others
- Interaction*fun*
```
flowchart LR
	A-->B
	B-->C
	C-->D
	click A callback "Tooltip for a callback"
	click B "http://www.github.com" "This is a tooltip for a link"
	click A call callback() "Tooltip for a callback"
	click B href "http://www.github.com" "This is a tooltip for a link"
```
- comments
```
%% this is a comment A -- text --> B{node}
```
- fontawesome
	- fa:fa-twitter
	- fa:fa-spinner
	- fa:fa-ban
	- fa:fa-camera-retro
```mermaid
flowchart LR
    B["fa:fa-twitter for peace"]
    B-->C[fa:fa-ban forbidden]
    B-->D(fa:fa-spinner);
    B-->E(A fa:fa-camera-retro perhaps?)

```
- 顶点和链接之间允许有一个空格。但是，顶点及其文本与链接及其文本之间不应有任何空格。

# Class diagram
# ER
# Others
## Sequence diagram
## Gantt diagram
## Pie
```mermaid
pie title Pets adopted by volunteers
    "Dogs" : 386
    "Cats" : 85
    "Rats" : 15
```
## User Journey
```mermaid
journey
    title My working day
    section Go to work
      Make tea: 5: Me
      Go upstairs: 3: Me
      Do work: 1: Me, Cat
    section Go home
      Go downstairs: 5: Me
      Sit down: 5: Me
```
#TD [[JavaScript]]