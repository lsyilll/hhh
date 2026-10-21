package local.guanye;
import java.util.*;
import org.springframework.jdbc.core.JdbcTemplate;
import jakarta.validation.Valid;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.jwt.Jwt;
import org.springframework.web.bind.annotation.*;
import org.springframework.transaction.annotation.Transactional;
@RestController @RequestMapping("/device") public class DeviceApi {
 final Workflow w;final JdbcTemplate db;public DeviceApi(Workflow w,JdbcTemplate db){this.w=w;this.db=db;}
 Map<String,Object> device(Jwt jwt){var d=w.row("devices",UUID.fromString(jwt.getSubject()));if(!"approved".equals(d.get("status")))throw new org.springframework.web.server.ResponseStatusException(org.springframework.http.HttpStatus.FORBIDDEN,"设备未获批准或已暂停");return d;}
 @GetMapping("/session") Object session(@AuthenticationPrincipal Jwt jwt){return device(jwt);}
 @PostMapping("/heartbeat") Object heartbeat(@AuthenticationPrincipal Jwt jwt){var d=device(jwt);db.update("UPDATE devices SET last_seen=now() WHERE id=?",d.get("id"));return Map.of("ok",true);}
 @PostMapping("/sample") @Transactional Object sample(@AuthenticationPrincipal Jwt jwt,@Valid @RequestBody Api.Sample s){var d=device(jwt);var project=(UUID)d.get("project_id");if(!project.equals(s.project_id())||!s.object_key().startsWith("users/"+d.get("owner_id")+"/"))throw new org.springframework.web.server.ResponseStatusException(org.springframework.http.HttpStatus.FORBIDDEN,"设备样本归属不正确");var id=UUID.randomUUID();int inserted=db.update("INSERT INTO platform_samples(id,project_id,owner_id,source_id,object_key,sha256,quality) VALUES(?,?,?,?,?,?,?::jsonb) ON CONFLICT(project_id,source_id) DO NOTHING",id,project,d.get("owner_id"),s.source_id(),s.object_key(),s.sha256(),w.encode(s.quality()));if(inserted==1){var task=UUID.randomUUID();db.update("INSERT INTO tasks(id,project_id,owner_id,kind,args) VALUES(?,?,?,'quality',?::jsonb)",task,project,d.get("owner_id"),w.encode(Map.of("sample_ids",List.of(id.toString()))));w.event(project,d.get("owner_id").toString(),"task.created",Map.of("task_id",task.toString(),"kind","quality"));w.event(project,d.get("owner_id").toString(),"sample.captured",Map.of("sample_id",id.toString(),"device_id",d.get("id").toString()));}return Map.of("accepted",true,"duplicate",inserted==0);}
}
