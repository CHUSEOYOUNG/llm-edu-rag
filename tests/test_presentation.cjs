const assert = require('node:assert/strict');
const test = require('node:test');
const guide = require('../web/presentation.js');

test('audiences change question suggestions without changing their scope', () => {
  const teacher = guide.examplesFor('teacher');
  const parent = guide.examplesFor('parent');
  assert.deepEqual(teacher.map(q => q.id), ['bytes', 'correction', 'hours']);
  assert.deepEqual(parent.map(q => q.id), ['correction', 'korean', 'date']);
  assert.equal(teacher[1].question, parent[0].question);
  assert.match(parent[2].question, /고시 제2026-1호/);
  assert.match(parent[2].question, /2028년 3월 1일/);
  assert.match(parent[2].question, /초등학교 1·2학년/);
  assert.deepEqual(guide.examplesFor('missing'), guide.examplesFor('all'));
  teacher[0].question = 'changed';
  assert.notEqual(guide.examplesFor('teacher')[0].question, 'changed');
});

test('filenames become readable titles without changing the source object', () => {
  assert.equal(guide.displayTitle('2026 학교생활기록부 기재요령(중)_F_260227'), '2026 학교생활기록부 기재요령 · 중학교');
  assert.equal(guide.displayTitle('2026 학교생활기록부 기재요령(초)_F_260219'), '2026 학교생활기록부 기재요령 · 초등학교');
  const notice = '교육과정_ 국가교육위원회 고시 제2026-1호(2026.1.21.)';
  assert.match(guide.displayTitle(notice), /제2026-1호\(2026.1.21.\)/);
  assert.equal(guide.topicFor('2026 학교생활기록부 기재요령'), '학교생활기록부');
  assert.equal(guide.topicFor('기타 자료'), '교육 자료');
});

test('school helpers detect one explicit level and label source documents', () => {
  assert.equal(guide.detectedSchoolLevel('중학교 출결이 궁금해요'), 'middle');
  assert.equal(guide.detectedSchoolLevel('초등학교와 중학교 수업 시간'), 'all');
  assert.equal(guide.detectedSchoolLevel('출결'), 'all');
  assert.deepEqual(guide.schoolLevelFor('기재요령(고)_F_1'), {value: 'high', label: '고등학교'});
  assert.deepEqual(guide.schoolLevelFor('교육과정 총론'), {value: 'all', label: '공통 자료'});
  assert.equal(guide.sectionTitle('학교생활 > 8조 출결상황'), '8조 출결상황');
});

test('highlight terms stay concise and similar source sections are grouped stably', () => {
  assert.deepEqual(guide.highlightTerms('중학교에서는 출결은 어떻게 처리하나요?'), ['중학교', '출결', '처리']);
  const sources = [
    {source_id: 'S1', doc_id: '문서', path: '출결'},
    {source_id: 'S2', doc_id: '문서', path: '출결'},
    {source_id: 'S3', doc_id: '문서', path: '학적'},
  ];
  const groups = guide.groupSources(sources);
  assert.equal(groups.length, 2);
  assert.deepEqual(groups[0].sources.map(source => source.source_id), ['S1', 'S2']);
  assert.deepEqual(groups[1].sources.map(source => source.source_id), ['S3']);
  assert.equal(sources[0].group, undefined);
});

test('plain-language condition labels distinguish literal locations without claiming applicability', () => {
  assert.deepEqual(guide.conditionLabel(['body', 'path']), {text: '내용에서 찾았어요', tone: 'found'});
  assert.deepEqual(guide.conditionLabel(['path']), {text: '항목 이름에서 찾았어요', tone: 'metadata'});
  assert.deepEqual(guide.conditionLabel(['doc_id']), {text: '자료 이름에서 찾았어요', tone: 'metadata'});
  assert.deepEqual(guide.conditionLabel([]), {text: '이 내용에서는 찾지 못했어요', tone: 'absent'});
});

test('PDF pages use reader-friendly labels in cards and saved notes', () => {
  assert.equal(guide.pageLabel({page_start: 7, page_end: 7}), '7쪽');
  assert.equal(guide.pageLabel({page_start: 7, page_end: 9}), '7~9쪽');
  assert.equal(guide.pageLabel({}), '페이지 정보가 없어요');
  assert.match(guide.sourceText({doc_id: '문서', path: '위치', body: '본문', page_start: 3, page_end: 4}, 1), /원문 페이지: 3~4쪽/);
});

test('HTML markers are removed and markdown tables become readable blocks', () => {
  const raw = '<!-- Start of picture text -->\n|항목|내용|\n|---|---|\n|글자|한글<br>3Byte<sup>1</sup>|\n<!-- End of picture text -->';
  assert.equal(guide.readableText(raw), '|항목|내용|\n|---|---|\n|글자|한글\n3Byte1|');
  assert.deepEqual(guide.readableBlocks(raw), [{type: 'table', headers: ['항목', '내용'], rows: [['글자', '한글\n3Byte1']]}]);
  assert.equal(guide.readablePreview(raw), '항목 · 내용\n글자 · 한글\n3Byte1');
  assert(!guide.readableDocument(raw).includes('|---|'));
  assert(!guide.readableText('본문<img src=x onerror=alert(1)>').includes('onerror'));
});

test('saved notes use readable content and exclude developer identifiers and scores', () => {
  const source = {doc_id: '2026 학교생활기록부 기재요령(중)_F_260227', path: '입력 안내',
    body: '한글 1자는 3Byte\n<img src=x onerror=alert(1)>', chunk_id: 'internal_chunk_id', score: 0.123456};
  const data = {missing_date_conditions: ['2028년 3월 1일'],
    context: {original_question: '질문 내용', sources: [source], omitted_chunk_ids: ['hidden-id']}};
  const saved = guide.saveText(data);
  assert(saved.includes('한글 1자는 3Byte'));
  assert(!saved.includes('<img'));
  assert(saved.includes(source.doc_id));
  assert(saved.includes('질문: 질문 내용'));
  assert(saved.includes('확인이 필요한 날짜: 2028년 3월 1일'));
  assert(saved.includes('표시하지 못한 자료'));
  assert(saved.includes('자동으로 작성한 답변은 아닙니다'));
  assert(!/internal_chunk_id|hidden-id|0\.123456|chunk_id|JSON|유사도/.test(saved));
});
