package local.guanye;
import java.util.*;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.security.oauth2.jwt.Jwt;
import org.springframework.transaction.annotation.Transactional;
import com.fasterxml.jackson.core.JsonProcessingException;
import tools.jackson.databind.ObjectMapper;
@Service public class Workflow {
 private final JdbcTemplate db; private final ObjectMapper json;
 public Workflow(JdbcTemplate db,ObjectMapper json){this.db=db;this.json=json;}
 String encode(Object value){return json.writeValueAsString(value);}
 static String uid(Jwt jwt){return jwt.getSubject();}
 public String role(UUID project,Jwt jwt){
  var rows=db.queryForList("SELECT role FROM memberships WHERE project_id=? AND user_id=?",project,uid(jwt));
  if(rows.isEmpty())throw new ResponseStatusException(HttpStatus.NOT_FOUND,"项目不存在或没有访问权限");return rows.getFirst().get("role").toString();
 }
 public void require(UUID project,Jwt jwt,String... roles){if(!Arrays.asList(roles).contains(role(project,jwt)))throw new ResponseStatusException(HttpStatus.FORBIDDEN,"项目角色不允许此操作");}
 public void audit(UUID p,Jwt jwt,String action,Object id,Object detail){db.update("INSERT INTO audit_log(project_id,actor_id,action,target_id,detail) VALUES(?,?,?,?,?::jsonb)",p,uid(jwt),action,id.toString(),encode(detail));}
 public void event(UUID p,String owner,String kind,Map<String,Object> payload){db.update("INSERT INTO outbox VALUES(?,?,?,?,?::jsonb,now(),false,false)",UUID.randomUUID(),kind,owner,p,encode(payload));}
 @Transactional public Map<String,Object> project(Jwt jwt,String name,String crop){var id=UUID.randomUUID();db.update("INSERT INTO projects(id,owner_id,name,crop) VALUES(?,?,?,?)",id,uid(jwt),name,crop);db.update("INSERT INTO memberships VALUES(?,?,?)",id,uid(jwt),"ADMIN");audit(id,jwt,"project.created",id,Map.of());return Map.of("id",id,"name",name);}
 @Transactional public Map<String,Object> device(Jwt jwt,UUID project,String name,String kind){require(project,jwt,"ADMIN","COLLECTOR");var id=UUID.randomUUID();db.update("INSERT INTO devices(id,project_id,owner_id,name,kind) VALUES(?,?,?,?,?)",id,project,uid(jwt),name,kind);audit(project,jwt,"device.requested",id,Map.of("kind",kind));return Map.of("id",id,"status","pending");}
 public Map<String,Object> row(String table,UUID id){if(!Set.of("devices","platform_samples","tasks").contains(table))throw new IllegalArgumentException();var rows=db.queryForList("SELECT * FROM "+table+" WHERE id=?",id);if(rows.isEmpty())throw new ResponseStatusException(HttpStatus.NOT_FOUND,"记录不存在");return rows.getFirst();}
 @Transactional public void deviceStatus(Jwt jwt,UUID id,String status){var d=row("devices",id);UUID p=(UUID)d.get("project_id");require(p,jwt,"ADMIN");db.update("UPDATE devices SET status=? WHERE id=?",status,id);audit(p,jwt,"device."+status,id,Map.of());event(p,uid(jwt),"device.status",Map.of("device_id",id.toString(),"status",status));}
 @Transactional public Map<String,Object> sample(Jwt jwt,UUID project,String source,String object,String sha,Map<String,Object> quality,String answer){require(project,jwt,"ADMIN","COLLECTOR");if(!object.startsWith("users/"+uid(jwt)+"/"))throw new ResponseStatusException(HttpStatus.BAD_REQUEST,"对象存储路径不属于当前账号");var id=UUID.randomUUID();db.update("INSERT INTO platform_samples(id,project_id,owner_id,source_id,object_key,sha256,quality,model_answer) VALUES(?,?,?,?,?,?,?::jsonb,?) ON CONFLICT(project_id,source_id) DO NOTHING",id,project,uid(jwt),source,object,sha,encode(quality),answer);var actual=db.queryForList("SELECT id FROM platform_samples WHERE project_id=? AND source_id=?",project,source).getFirst().get("id");audit(project,jwt,"sample.imported",actual,Map.of("source_id",source));event(project,uid(jwt),"sample.imported",Map.of("sample_id",actual.toString()));return Map.of("id",actual);}
 @Transactional public void review(Jwt jwt,UUID id,String status,String label,String notes){var s=row("platform_samples",id);var p=(UUID)s.get("project_id");require(p,jwt,"ADMIN","REVIEWER");if(status.equals("confirmed")&&label.isBlank())throw new ResponseStatusException(HttpStatus.BAD_REQUEST,"确认前请填写人工标签");db.update("UPDATE platform_samples SET review_status=?,label=?,notes=?,reviewer=? WHERE id=?",status,label,notes,uid(jwt),id);db.update("INSERT INTO reviews VALUES(?,?,?,?,?,?,now())",UUID.randomUUID(),id,uid(jwt),status,label,notes);audit(p,jwt,"sample.reviewed",id,Map.of("status",status,"label",label));event(p,s.get("owner_id").toString(),"sample.reviewed",Map.of("sample_id",id.toString(),"status",status,"label",label));}
 @Transactional public Map<String,Object> task(Jwt jwt,UUID project,String kind,Map<String,Object> args){require(project,jwt,"ADMIN","COLLECTOR","REVIEWER");var ids=args.get("sample_ids");if(ids instanceof List<?> list){if(list.size()>100)throw new ResponseStatusException(HttpStatus.BAD_REQUEST,"每批最多100个样本");for(Object id:list){var s=row("platform_samples",UUID.fromString(id.toString()));if(!s.get("project_id").equals(project))throw new ResponseStatusException(HttpStatus.FORBIDDEN,"样本不属于该项目");}}var id=UUID.randomUUID();db.update("INSERT INTO tasks(id,project_id,owner_id,kind,args) VALUES(?,?,?,?,?::jsonb)",id,project,uid(jwt),kind,encode(args));event(project,uid(jwt),"task.created",Map.of("task_id",id.toString(),"kind",kind));audit(project,jwt,"task.created",id,Map.of("kind",kind));return Map.of("id",id,"status","queued");}
}
