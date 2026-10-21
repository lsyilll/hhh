package local.guanye;
import java.util.*;
import org.springframework.jdbc.core.JdbcTemplate;
import tools.jackson.databind.ObjectMapper;
import jakarta.validation.Valid;
import jakarta.validation.constraints.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.jwt.Jwt;
import org.springframework.transaction.annotation.Transactional;
@RestController @RequestMapping("/v1") public class Api {
 final Workflow w;final JdbcTemplate db;final ObjectMapper json;public Api(Workflow w,JdbcTemplate db,ObjectMapper json){this.w=w;this.db=db;this.json=json;}
 record Project(@NotBlank @Size(max=100) String name,@Size(max=80) String crop){}
 record Device(@NotNull UUID project_id,@NotBlank @Size(max=100) String name,@NotNull @Pattern(regexp="phone|camera|raspberrypi") String kind){}
 record Status(@NotNull @Pattern(regexp="approved|paused|rejected") String status){}
 record Sample(@NotNull UUID project_id,@NotBlank @Size(max=64) String source_id,@NotBlank @Size(max=300) String object_key,@NotNull @Pattern(regexp="[a-f0-9]{64}") String sha256,@NotNull Map<String,Object> quality,@Size(max=50000) String model_answer){}
 record Review(@NotNull @Pattern(regexp="pending|confirmed|uncertain|rephoto") String status,@NotNull @Size(max=100) String label,@NotNull @Size(max=4000) String notes){}
 record Task(@NotNull UUID project_id,@NotNull @Pattern(regexp="quality|features|dinov3|segment|evidence_report|dataset_audit|experiment") String kind,@NotNull Map<String,Object> args){}
 record Member(@NotBlank @Size(max=64) String user_id,@NotNull @Pattern(regexp="COLLECTOR|REVIEWER") String role){}
 @GetMapping("/projects") Object projects(@AuthenticationPrincipal Jwt j){return db.queryForList("SELECT p.*,m.role FROM projects p JOIN memberships m ON m.project_id=p.id WHERE m.user_id=? ORDER BY p.created DESC",j.getSubject());}
 @PostMapping("/projects") Object create(@AuthenticationPrincipal Jwt j,@Valid @RequestBody Project p){return w.project(j,p.name,p.crop==null?"":p.crop);}
 @GetMapping("/projects/{id}/members") Object members(@AuthenticationPrincipal Jwt j,@PathVariable UUID id){w.require(id,j,"ADMIN");return db.queryForList("SELECT user_id,role FROM memberships WHERE project_id=?",id);}
 @PostMapping("/projects/{id}/members") @Transactional Object member(@AuthenticationPrincipal Jwt j,@PathVariable UUID id,@Valid @RequestBody Member m){w.require(id,j,"ADMIN");if(m.user_id.equals(j.getSubject()))throw new org.springframework.web.server.ResponseStatusException(org.springframework.http.HttpStatus.BAD_REQUEST,"不能通过此入口降级自己的管理员角色");db.update("INSERT INTO memberships VALUES(?,?,?) ON CONFLICT(project_id,user_id) DO UPDATE SET role=excluded.role",id,m.user_id,m.role);w.audit(id,j,"member.granted",m.user_id,Map.of("role",m.role));return Map.of("saved",true);}
 @GetMapping("/devices") Object devices(@AuthenticationPrincipal Jwt j){return db.queryForList("SELECT d.* FROM devices d JOIN memberships m ON m.project_id=d.project_id WHERE m.user_id=? ORDER BY d.created DESC",j.getSubject());}
 @PostMapping("/devices") Object createDevice(@AuthenticationPrincipal Jwt j,@Valid @RequestBody Device d){return w.device(j,d.project_id,d.name,d.kind);}
 @PatchMapping("/devices/{id}") Object deviceStatus(@AuthenticationPrincipal Jwt j,@PathVariable UUID id,@Valid @RequestBody Status s){w.deviceStatus(j,id,s.status);return Map.of("saved",true);}
 @GetMapping("/devices/{id}") Object device(@AuthenticationPrincipal Jwt j,@PathVariable UUID id){var d=w.row("devices",id);w.role((UUID)d.get("project_id"),j);return d;}
 @GetMapping("/samples") Object samples(@AuthenticationPrincipal Jwt j,@RequestParam UUID project_id){w.role(project_id,j);return db.queryForList("SELECT * FROM platform_samples WHERE project_id=? ORDER BY created DESC LIMIT 200",project_id);}
 @GetMapping("/samples/{id}") Object sample(@AuthenticationPrincipal Jwt j,@PathVariable UUID id){var s=w.row("platform_samples",id);w.role((UUID)s.get("project_id"),j);return s;}
 @PostMapping("/samples") Object createSample(@AuthenticationPrincipal Jwt j,@Valid @RequestBody Sample s){return w.sample(j,s.project_id,s.source_id,s.object_key,s.sha256,s.quality,s.model_answer==null?"":s.model_answer);}
 @PostMapping("/samples/{id}/review") Object review(@AuthenticationPrincipal Jwt j,@PathVariable UUID id,@Valid @RequestBody Review r){w.review(j,id,r.status,r.label,r.notes);return Map.of("saved",true);}
 @GetMapping("/tasks") Object tasks(@AuthenticationPrincipal Jwt j){return db.queryForList("SELECT t.* FROM tasks t JOIN memberships m ON m.project_id=t.project_id WHERE m.user_id=? ORDER BY t.created DESC LIMIT 100",j.getSubject());}
 @GetMapping("/tasks/{id}") Object task(@AuthenticationPrincipal Jwt j,@PathVariable UUID id){var t=w.row("tasks",id);w.role((UUID)t.get("project_id"),j);return t;}
 @PostMapping("/tasks") Object createTask(@AuthenticationPrincipal Jwt j,@Valid @RequestBody Task t){return w.task(j,t.project_id,t.kind,t.args);}
 @PostMapping("/tasks/{id}/retry") Object retry(@AuthenticationPrincipal Jwt j,@PathVariable UUID id){var t=w.row("tasks",id);w.require((UUID)t.get("project_id"),j,"ADMIN","COLLECTOR","REVIEWER");if(!Set.of("failed","cancelled").contains(t.get("status")))throw new org.springframework.web.server.ResponseStatusException(org.springframework.http.HttpStatus.CONFLICT,"只能重试失败或已取消的任务");return w.task(j,(UUID)t.get("project_id"),t.get("kind").toString(),json.readValue(t.get("args").toString(),Map.class));}
 @GetMapping("/experiments") Object experiments(@AuthenticationPrincipal Jwt j){return db.queryForList("SELECT e.* FROM model_experiments e JOIN memberships m ON m.project_id=e.project_id WHERE m.user_id=? ORDER BY e.created DESC LIMIT 100",j.getSubject());}
 @GetMapping("/datasets") Object datasets(@AuthenticationPrincipal Jwt j){return db.queryForList("SELECT e.* FROM dataset_versions e JOIN memberships m ON m.project_id=e.project_id WHERE m.user_id=? ORDER BY e.created DESC LIMIT 100",j.getSubject());}
 @GetMapping("/audit") Object audit(@AuthenticationPrincipal Jwt j,@RequestParam UUID project_id){w.require(project_id,j,"ADMIN","REVIEWER");return db.queryForList("SELECT actor_id,action,target_id,detail,created FROM audit_log WHERE project_id=? ORDER BY id DESC LIMIT 200",project_id);}
}
