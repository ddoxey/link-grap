# link-graph

Produce a graph image for link-parser sentence diagram.

Requirements:

    * link-5.12.5 installed in the user's home dir
    * write access to Desktop directory in the user's home dir

Usage:

    parse.py My hovercraft is full of eels.

```
    +----------------------Xp---------------------+
    +----------->WV---------->+                   |
    +------>Wd-------+        |                   |
    |       +--Ds**c-+--Ss*s--+--Pa-+-OFj+-Jp-+   |
    |       |        |        |     |    |    |   |
LEFT-WALL my.p hovercraft.s is.v full.a of eels.n .
```

The result will be link-parser.png in the user's Desktop directory.

![My hovercraft is full of eels.](https://github.com/ddoxey/link-graph/blob/main/link-parser.png)
