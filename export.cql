call apoc.export.csv.query("match (n) return id(n) as nid,n.id as id,n.name as name,labels(n)[0] as lab", "../../graph-embedding/data/entity.csv", {});
call apoc.export.csv.query("match (n)-[r]->(g) return id(n) as hid,n.id as nid,type(r) as rel,g.id as gid,id(g) as tid,n.name as nname,g.name as gname", "../../graph-embedding/data/triple.csv", {});
