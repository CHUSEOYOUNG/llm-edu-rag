"use strict";

// Presentation only: never changes the search index or adds hidden query filters.
const schoolGuide = (() => {
  const questions = {
    bytes: {label: "학교생활기록부 글자 수는 어떻게 세나요?", question: "생기부 글자수 셀 때 한글 한 글자는 몇 바이트로 계산되나요?", topic: "학교생활기록부"},
    correction: {label: "지난해 학교생활기록부를 고칠 수 있나요?", question: "작년 생기부에 잘못 쓴 내용이 있는데 지금 고칠 수 있나요?", topic: "학교생활기록부"},
    hours: {label: "학교에서 수업 시간을 조정할 수 있나요?", question: "초등학교에서 교과 수업시수를 학교 재량으로 줄일 수 있나요?", topic: "수업과 교육과정"},
    duration: {label: "초등학교와 중학교 수업은 몇 분인가요?", question: "2022 개정 교육과정 기준으로 초등학교와 중학교의 수업 한 시간은 각각 몇 분이 원칙인가요?", topic: "수업과 교육과정"},
    electives: {label: "중학교에서는 어떤 과목을 선택하나요?", question: "중학교 선택 교과에는 어떤 과목들이 있나요?", topic: "배우는 과목"},
    information: {label: "초등학교와 중학교의 정보 교육은 얼마나 하나요?", question: "2022 개정 교육과정 기준으로 초등학교와 중학교의 정보 교육은 각각 최소 몇 시간을 편성해야 하나요?", topic: "배우는 과목"},
    korean: {label: "초등학교 3~4학년 국어 수업은 얼마나 하나요?", question: "초등학교 3~4학년 국어 수업 시수는 몇 시간인가요?", topic: "배우는 과목"},
    date: {label: "2028년 초등학교 1~2학년은 무엇을 배우나요?", question: "국가교육위원회 고시 제2026-1호에 따라, 2028년 3월 1일부터 초등학교 1·2학년이 배우는 교과는 무엇인가요?", topic: "배우는 과목"}
  };
  const audiences = {
    all: ["duration", "correction", "electives"],
    teacher: ["bytes", "correction", "hours"],
    student: ["duration", "electives", "information"],
    parent: ["correction", "korean", "date"]
  };

  function examplesFor(audience) {
    return (audiences[audience] || audiences.all).map((id) => ({id, ...questions[id]}));
  }

  function displayTitle(name) {
    return name.replace(/_F_\d+$/, "")
      .replace(/기재요령\(초\)/g, "기재요령 · 초등학교")
      .replace(/기재요령\(중\)/g, "기재요령 · 중학교")
      .replace(/기재요령\(고\)/g, "기재요령 · 고등학교")
      .replace(/_/g, " · ");
  }

  function topicFor(name) {
    if (name.includes("학교생활기록부")) return "학교생활기록부";
    if (name.includes("교육과정")) return "수업과 교육과정";
    return "교육 자료";
  }

  function conditionLabel(fields) {
    if (fields.includes("body")) return {text: "내용에서 찾았어요", tone: "found"};
    if (fields.includes("path")) return {text: "항목 이름에서 찾았어요", tone: "metadata"};
    if (fields.includes("doc_id")) return {text: "자료 이름에서 찾았어요", tone: "metadata"};
    return {text: "이 내용에서는 찾지 못했어요", tone: "absent"};
  }

  function pageLabel(source) {
    const start = Number(source.page_start);
    const end = Number(source.page_end);
    if (!Number.isInteger(start) || start < 1) return "페이지 정보가 없어요";
    return Number.isInteger(end) && end > start ? `${start}~${end}쪽` : `${start}쪽`;
  }

  function sourceText(source, number) {
    return [`관련 자료 ${number}`, `자료 이름: ${source.doc_id}`, `원문 페이지: ${pageLabel(source)}`, `자료 안의 위치: ${source.path || "위치 정보가 없어요"}`, "", source.body].join("\n");
  }

  function saveText(data) {
    const notice = "아래 내용은 질문과 관련해 찾은 자료의 일부입니다. 학교급·학년·적용 날짜가 내 상황과 맞는지 확인해 주세요. 자동으로 작성한 답변은 아닙니다.";
    const dates = data.missing_date_conditions.length ? `\n확인이 필요한 날짜: ${data.missing_date_conditions.join(", ")}\n` : "";
    const omitted = data.context.omitted_chunk_ids.length ? "\n내용이 길어 이번에 표시하지 못한 자료도 있습니다.\n" : "";
    return ["학교생활 안내 · 찾아본 내용", `질문: ${data.context.original_question}`, "", notice, dates, omitted,
      ...data.context.sources.map((source, index) => "────────────────────────\n" + sourceText(source, index+1))].join("\n") + "\n";
  }

  return {examplesFor, displayTitle, topicFor, conditionLabel, pageLabel, sourceText, saveText};
})();

if (typeof module !== "undefined") module.exports = schoolGuide;
