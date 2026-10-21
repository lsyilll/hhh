import{test,expect}from'@playwright/test';
test.beforeEach(async({page})=>{await page.route('**/api/**',async route=>{const url=new URL(route.request().url());let data:unknown={detail:'Integration fixture: service unavailable'};let status=503;if(url.pathname==='/api/operations'){status=200;data={samples:0,review_counts:{pending:0},case_count:0,average_seconds:null,trend:[],accuracy_note:'尚未建立独立评测集'};}if(url.pathname==='/api/platform/health'){status=200;data={services:[],docker_available:false};}if(url.pathname==='/api/platform/catalog'){status=200;data={technologies:[{name:'DINOv3 ViT-B/16',module:'models',purpose:'视觉特征与案例检索',condition:'官方授权及权重待本人完成'}]};}if(url.pathname==='/api/platform/events'){await route.fulfill({status:200,contentType:'text/event-stream',body:': fixture\n\n'});return;}await route.fulfill({status,contentType:'application/json',body:JSON.stringify(data)});});});
test('shows actual empty data and unavailable-service states',async({page})=>{await page.goto('/platform/');await expect(page.getByText('让每个判断，都留下依据。')).toBeVisible();await expect(page.getByText('完整服务需要 Docker / WSL2')).toBeVisible();await page.getByText('项目与权限',{exact:true}).click();await expect(page.getByText('服务尚未就绪')).toBeVisible();});
test('technical assignment is readable on mobile without horizontal overflow',async({page})=>{await page.setViewportSize({width:390,height:844});await page.goto('/platform/#stack');await expect(page.getByText('技术栈与运行边界',{exact:true})).toBeVisible();expect(await page.evaluate(()=>document.documentElement.scrollWidth<=document.documentElement.clientWidth)).toBeTruthy();});

test('public SAM2 requires a selected sample and box, without a text prompt',async({page})=>{
 let submitted:any;
 const project='00000000-0000-0000-0000-000000000001';const sample='00000000-0000-0000-0000-000000000002';
 await page.route('**/api/platform/business/**',async route=>{
  const path=new URL(route.request().url()).pathname;
  let data:unknown=[];let status=200;
  if(path.endsWith('/projects'))data=[{id:project,name:'自动测试项目',role:'ADMIN'}];
  if(path.endsWith('/samples'))data=[{id:sample,source_id:'浏览器用例样本'}];
  if(path.endsWith('/tasks')&&route.request().method()==='POST'){submitted=route.request().postDataJSON();status=503;data={detail:'UI test only: no production task submitted'};}
  await route.fulfill({status,contentType:'application/json',body:JSON.stringify(data)});
 });
 await page.route('**/api/platform/object?**',route=>route.fulfill({contentType:'image/png',body:Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a7f8AAAAASUVORK5CYII=','base64')}));
 await page.goto('/platform/#lab');
 await expect(page.getByText('DINOv2 ViT-B/14 · 公开权重',{exact:true})).toBeVisible();
 await expect(page.getByRole('button',{name:'生成分割候选',exact:true})).toBeDisabled();
 const picker=page.getByRole('combobox',{name:'待分割样本',exact:true});await picker.click();await picker.press('ArrowDown');await picker.press('Enter');
 await page.getByRole('spinbutton',{name:'分割框中心 X',exact:true}).fill('0.5');
 await page.getByRole('spinbutton',{name:'分割框宽度',exact:true}).fill('0.9');
 await page.getByRole('spinbutton',{name:'分割框高度',exact:true}).fill('0.9');
 await page.getByRole('spinbutton',{name:'分割框高度',exact:true}).press('Tab');
 await page.getByRole('button',{name:'生成分割候选',exact:true}).click();
 await expect.poll(()=>submitted).toMatchObject({kind:'segment',args:{segmenter:'sam2',sample_ids:[sample],box:[.5,.5,.9,.9]}});
 expect(submitted.args).not.toHaveProperty('prompt');
});
