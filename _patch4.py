p = 'main.py'
s = open(p, encoding='utf-8', newline='').read()
nl = chr(10)
def J(x): return nl.join(x)

old_imp = 'from slowapi.errors import RateLimitExceeded'
assert s.count(old_imp) == 1
s = s.replace(old_imp, old_imp + nl + 'from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials', 1)

sig_reps = [
 (J(['    user_id: str = Form(...)', ')']),
  J(['    uid: str = Depends(resolve_user_id)', ')']), 2),
 (J(['    user_id: str = Form(...),', '    image_method']),
  J(['    uid: str = Depends(resolve_user_id),', '    image_method']), 1),
 (J(['    user_id: str = Form(...),', '    max_posts: int = Form(5)']),
  J(['    uid: str = Depends(resolve_user_id),', '    max_posts: int = Form(5)']), 1),
 (J(['    user_id: str = Form(...),', '    db: Session = Depends(get_db)']),
  J(['    uid: str = Depends(resolve_user_id),', '    db: Session = Depends(get_db)']), 1),
 (J(['    user_id: str = Form(...),', '    discord_webhook_url: Optional[str] = Form(None),']),
  J(['    uid: str = Depends(resolve_user_id),', '    discord_webhook_url: Optional[str] = Form(None),']), 1),
]
for o, n, cnt in sig_reps:
    c = s.count(o)
    assert c == cnt, 'SIG COUNT %d != %d: %r' % (c, cnt, o[:70])
    s = s.replace(o, n)

body_reps = [
 ('return await _generate_post_core(audio_file, tone, platform, user_id)',
  'return await _generate_post_core(audio_file, tone, platform, uid)', 1),
 ('post_response = await _generate_post_core(audio_file, tone, platform, user_id)',
  'post_response = await _generate_post_core(audio_file, tone, platform, uid)', 1),
 ('results = vector_store.search_index(transcript, top_k=3, user_id=user_id)',
  'results = vector_store.search_index(transcript, top_k=3, user_id=uid)', 2),
 ('creds = db.query(SocialCreds).filter(SocialCreds.user_id == user_id).first()',
  'creds = db.query(SocialCreds).filter(SocialCreds.user_id == uid).first()', 2),
 ('vector_store.add_text_to_index([memory_text], user_id=user_id)',
  'vector_store.add_text_to_index([memory_text], user_id=uid)', 1),
]
for o, n, cnt in body_reps:
    c = s.count(o)
    assert c == cnt, 'BODY COUNT %d != %d: %r' % (c, cnt, o[:70])
    s = s.replace(o, n)

anchor = '# ==================== Bio Syncing Helpers ===================='
assert s.count(anchor) == 1
block = nl.join([
'# ==================== Lightweight Auth for Mobile Clients ====================',
'_bearer_optional = HTTPBearer(auto_error=False)',
'',
'async def resolve_user_id(',
'    request: Request,',
'    user_id: Optional[str] = Form(None),',
'    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_optional),',
'    db: Session = Depends(get_db),',
') -> str:',
'    """',
'    Resolve the acting user securely but friction-free:',
'      1. Authorization: Bearer JWT   (preferred - issued silently by /auth/device)',
'      2. X-API-Key header            (for server-to-server clients)',
'      3. Plain user_id form field    (development/testing only)',
'    Layman users never see any of this - the app handles it automatically.',
'    """',
'    if credentials is not None:',
'        payload = auth_service.decode_access_token(credentials.credentials)',
'        uid = payload.get("sub")',
'        if not uid:',
'            raise HTTPException(status_code=401, detail="Invalid token payload")',
'        return uid',
'',
'    api_key = request.headers.get("x-api-key")',
'    if api_key:',
'        uid = auth_service.verify_api_key(db, api_key)',
'        if not uid:',
'            raise HTTPException(status_code=401, detail="Invalid or expired API key")',
'        return uid',
'',
'    if user_id:',
'        if ENVIRONMENT == "production":',
'            raise HTTPException(',
'                status_code=401,',
'                detail="Authentication required. Get a token from /auth/device or /auth/login "',
'                       "and send it as Authorization: Bearer token."',
'            )',
'        return user_id',
'',
'    raise HTTPException(status_code=401, detail="Authentication required.")',
'',
'',
'def _identity_matches(request: Request, db: Session, claimed_user_id: str) -> bool:',
'    """Check that the caller is authenticated AS the claimed user_id."""',
'    auth_header = request.headers.get("authorization") or ""',
'    if auth_header.lower().startswith("bearer "):',
'        try:',
'            payload = auth_service.decode_access_token(auth_header[7:].strip())',
'        except HTTPException:',
'            return False',
'        return payload.get("sub") == claimed_user_id',
'    api_key = request.headers.get("x-api-key")',
'    if api_key:',
'        return auth_service.verify_api_key(db, api_key) == claimed_user_id',
'    return False',
'',
'',
"@app.post('/auth/device')",
"@limiter.limit(RATE_LIMITS['auth'])",
'async def device_signup(request: Request, db: Session = Depends(get_db)):',
'    """ZERO-FRICTION signup for mobile apps.',
'    Call ONCE on first app launch, store user_id + access_token, then send',
'    Authorization: Bearer token on every request. The user never sees a login screen."""',
'    import uuid, secrets',
'    device_tag = uuid.uuid4().hex[:12]',
'    email = "device_" + device_tag + "@app.local"',
'    password = secrets.token_urlsafe(24)',
'    try:',
'        user = auth_service.create_user(db, email, password)',
'        upload_db()',
'        token = auth_service.create_access_token({"sub": user.user_id})',
'        return {',
'            "status": "success",',
'            "user_id": user.user_id,',
'            "access_token": token,',
'            "token_type": "bearer",',
'            "message": "Store these credentials securely and reuse them on every request."',
'        }',
'    except HTTPException:',
'        raise',
'    except Exception as e:',
'        raise HTTPException(status_code=500, detail=str(e))',
'',
'',
'', anchor])
s = s.replace(anchor, block, 1)

c_anchor = J([
 'async def confirm_post(request: Request, post_request: ConfirmPostRequest, db: Session = Depends(get_db)):',
 '    if not post_request.scheduled_time:'])
c_new = J([
 'async def confirm_post(request: Request, post_request: ConfirmPostRequest, db: Session = Depends(get_db)):',
 '    # Safeguard: in production the caller must be authenticated AS this user_id',
 '    if ENVIRONMENT == "production" and not _identity_matches(request, db, post_request.user_id):',
 '        raise HTTPException(status_code=401, detail="Authentication required or user_id mismatch.")',
 '    if not post_request.scheduled_time:'])
assert s.count(c_anchor) == 1
s = s.replace(c_anchor, c_new, 1)

open(p, 'w', encoding='utf-8', newline='').write(s)
print('AUTH SAFEGUARD APPLIED')

readme_add = nl.join([
'',
'---',
'',
'## Mobile Authentication (zero-friction)',
'',
'The API is protected so no one can act as another user - but the app feel stays hassle-free:',
'',
'1. On first app launch, call POST /auth/device (no body needed). It silently creates a',
'   private account and returns {"status":"success","user_id":"...","access_token":"..."}.',
'2. Store user_id + access_token in app storage.',
'3. On every request, send header: Authorization: Bearer access_token.',
'   You may still include user_id in form data, but the server always trusts the token.',
'',
'Users NEVER see a login screen. Email/password login (/auth/register, /auth/login) is',
'optional - only for users who want to sync across devices.',
'',
'In ENVIRONMENT=development the plain-user_id flow still works for quick testing;',
'in production it is rejected with 401.',
''])
with open('README.md', 'a', encoding='utf-8', newline='') as f:
    f.write(readme_add + nl)
print('README UPDATED')
