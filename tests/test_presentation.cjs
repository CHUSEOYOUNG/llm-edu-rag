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

test('plain-language condition labels distinguish literal locations without claiming applicability', () => {
  assert.deepEqual(guide.conditionLabel(['body', 'path']), {text: '내용에서 찾았어요', tone: 'found'});
  assert.deepEqual(guide.conditionLabel(['path']), {text: '항목 이름에서 찾았어요', tone: 'metadata'});
  assert.deepEqual(guide.conditionLabel(['doc_id']), {text: '자료 이름에서 찾았어요', tone: 'metadata'});
  assert.deepEqual(guide.conditionLabel([]), {text: '이 내용에서는 찾지 못했어요', tone: 'absent'});
});

test('saved notes preserve original content and citations but exclude developer identifiers and scores', () => {
  const source = {doc_id: '2026 학교생활기록부 기재요령(중)_F_260227', path: '입력 안내',
    body: '한글 1자는 3Byte\n<img src=x onerror=alert(1)>', chunk_id: 'internal_chunk_id', score: 0.123456};
  const data = {missing_date_conditions: ['2028년 3월 1일'],
    context: {original_question: '질문 내용', sources: [source], omitted_chunk_ids: ['hidden-id']}};
  const saved = guide.saveText(data);
  assert(saved.includes(source.body));
  assert(saved.includes(source.doc_id));
  assert(saved.includes('질문: 질문 내용'));
  assert(saved.includes('확인이 필요한 날짜: 2028년 3월 1일'));
  assert(saved.includes('표시하지 못한 자료'));
  assert(saved.includes('자동으로 작성한 답변은 아닙니다'));
  assert(!/internal_chunk_id|hidden-id|0\.123456|chunk_id|JSON|유사도/.test(saved));
});
