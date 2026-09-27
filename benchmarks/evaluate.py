"""Every declared row counts; missing, invalid and unparsed outputs remain failures."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random

PROBABILITY_NORMALIZATION_TOLERANCE = 1e-4
NLL_PROBABILITY_FLOOR = 1e-12
ANSWER_LETTERS = 'ABCDEFGHIJKLMNOP'
HARD_PROJECTION_FIELDS = ('label', 'target_option_id', 'target_letter')


def indexed(rows, name):
    result = {}
    for row in rows:
        if row['id'] in result:
            raise ValueError(f'Duplicate ID in {name}: {row["id"]}')
        result[row['id']] = row
    return result


def read_jsonl(path):
    with open(path) as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    indexed(rows, str(path))
    return rows


def vector(values, ids):
    if isinstance(values, dict):
        if set(values) != set(ids):
            raise ValueError('Probability keys differ from gold option IDs')
        values = [values[key] for key in ids]
    if not isinstance(values, list) or len(values) != len(ids):
        raise ValueError('Wrong probability vector length/type')
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p)
           or not 0 <= p <= 1 for p in values):
        raise ValueError('Invalid/nonfinite probability')
    if abs(sum(values)-1) > PROBABILITY_NORMALIZATION_TOLERANCE:
        raise ValueError('Probabilities do not sum to one')
    return values


def normalized_target(values):
    """Normalize a target accepted within the declared mass tolerance."""
    total = sum(values)
    return [value/total for value in values]


def top_1_in_answer_set(probabilities, ids, answer_set):
    """Score exact ties conservatively so option ordering cannot change this metric."""
    maximum = max(probabilities)
    return all(option_id in answer_set for option_id, probability in zip(ids, probabilities) if probability == maximum)


def _record_identity(item):
    return repr(item.get('id', '<missing id>'))


def _semantic_option_ids(item):
    """Return the IDs needed to validate a semantic answer set."""
    identity = _record_identity(item)
    options = item.get('options')
    if not isinstance(options, list):
        raise ValueError(f'Invalid answer-set record {identity}: options must be a list')
    ids = []
    for option in options:
        if not isinstance(option, dict) or not isinstance(option.get('id'), str) or not option['id'].strip():
            raise ValueError(f'Invalid answer-set record {identity}: options need nonempty string IDs')
        ids.append(option['id'])
    if len(ids) != len(set(ids)):
        raise ValueError(f'Invalid answer-set record {identity}: option IDs must be unique')
    return ids


def answer_set_contract(item):
    """Validate one optional soft-label record and return normalized gold data."""
    identity = _record_identity(item)
    ids = _semantic_option_ids(item)
    answer_set = item.get('answer_set')
    if not isinstance(answer_set, list) or not answer_set:
        raise ValueError(f'Invalid answer-set record {identity}: answer_set must be a nonempty list')
    if any(not isinstance(answer, str) or not answer.strip() for answer in answer_set):
        raise ValueError(f'Invalid answer-set record {identity}: answer_set IDs must be nonempty strings')
    if len(answer_set) != len(set(answer_set)):
        raise ValueError(f'Invalid answer-set record {identity}: answer_set IDs must be unique')
    unknown = set(answer_set)-set(ids)
    if unknown:
        raise ValueError(f'Invalid answer-set record {identity}: unknown answer_set IDs {sorted(unknown)}')
    if 'target_distribution' not in item or item['target_distribution'] is None:
        raise ValueError(f'Invalid answer-set record {identity}: target_distribution is required')
    try:
        target = vector(item['target_distribution'], ids)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid answer-set record {identity}: {exc}') from exc
    support = {option_id for option_id, probability in zip(ids, target) if probability > 0}
    if support != set(answer_set):
        raise ValueError(
            f'Invalid answer-set record {identity}: positive target_distribution support must equal answer_set'
        )
    projection_present = [field in item for field in HARD_PROJECTION_FIELDS]
    if len(answer_set) > 1:
        if any(projection_present):
            raise ValueError(f'Invalid answer-set record {identity}: multi-answer rows must omit hard projection fields')
        projection = None
    elif any(projection_present) and not all(projection_present):
        raise ValueError(f'Invalid answer-set record {identity}: singleton hard projection must be complete')
    elif all(projection_present):
        label = item['label']
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(ids):
            raise ValueError(f'Invalid answer-set record {identity}: label must be an in-range option index')
        if item['target_option_id'] != answer_set[0] or item['target_option_id'] != ids[label]:
            raise ValueError(f'Invalid answer-set record {identity}: target_option_id must match the singleton answer_set')
        if label >= len(ANSWER_LETTERS) or item['target_letter'] != ANSWER_LETTERS[label]:
            raise ValueError(f'Invalid answer-set record {identity}: target_letter must match the option index')
        projection = dict(label=label, target_option_id=item['target_option_id'], target_letter=item['target_letter'])
    else:
        projection = None
    return dict(answer_set=tuple(answer_set), target=normalized_target(target), hard_projection=projection)


def gold_contracts(gold):
    """Validate the explicit answer-set variant before predictions are inspected."""
    contracts = {}
    for item in gold:
        identity = _record_identity(item)
        if 'answer_set' in item:
            contracts[item['id']] = answer_set_contract(item)
        elif 'label' not in item:
            raise ValueError(f'Invalid gold record {identity}: requires a legacy label or answer_set')
    return contracts


def require_hard_label_gold(gold, context):
    """Allow only legacy labels and complete singleton hard projections."""
    contracts = gold_contracts(gold)
    unsupported = [record_id for record_id, contract in contracts.items() if contract['hard_projection'] is None]
    if unsupported:
        records = ', '.join(repr(record_id) for record_id in unsupported)
        raise ValueError(
            f'{context} requires legacy hard labels or complete singleton hard projections; '
            f'unsupported answer-set records: {records}'
        )


def align(gold, predictions):
    truth = indexed(gold, 'gold')
    contracts = gold_contracts(gold)
    outputs = indexed(predictions, 'predictions')
    if outputs.keys() - truth.keys():
        raise ValueError(f'Unknown prediction IDs: {sorted(outputs.keys()-truth.keys())[:5]}')
    rows = []
    for item in gold:
        ids = [o['id'] for o in item['options']]
        contract = contracts.get(item['id'])
        if contract is None:
            if len(set(ids)) != len(ids) or not 0 <= item['label'] < len(ids):
                raise ValueError(f'Invalid gold options/label: {item["id"]}')
            target = item.get('target_distribution')
            if target is not None:
                target = vector(target, ids)
            row = dict(id=item['id'], group_id=item['group_id'], family=item['family'],
                       gold_id=ids[item['label']], predicted_id=None, correct=False,
                       confidence=None, nll=None, brier=None, status='missing', probabilities=None)
        elif contract['hard_projection'] is not None:
            label = contract['hard_projection']['label']
            target = contract['target']
            row = dict(id=item['id'], group_id=item['group_id'], family=item['family'],
                       gold_id=ids[label], predicted_id=None, correct=False,
                       confidence=None, nll=None, brier=None, status='missing', probabilities=None)
            row.update(answer_set=list(contract['answer_set']), top_1_in_set=False,
                       expected_nll=None, expected_brier=None,
                       acceptable_set_probability_mass=None)
        else:
            target = contract['target']
            row = dict(id=item['id'], group_id=item['group_id'], family=item['family'],
                       answer_set=list(contract['answer_set']), predicted_id=None, correct=False,
                       top_1_in_set=False, confidence=None, expected_nll=None,
                       expected_brier=None, acceptable_set_probability_mass=None,
                       status='missing', probabilities=None)
        pred = outputs.get(item['id'])
        if pred is not None:
            try:
                if pred.get('parse_status') == 'unparsed' or pred.get('error'):
                    raise ValueError(pred.get('parse_error') or pred.get('error') or 'Unparsed output')
                values = pred.get('probabilities')
                if values is None:
                    choice = pred.get('prediction_id')
                    if choice not in ids:
                        raise ValueError('Missing or out-of-set native prediction')
                    row.update(status='native_decision', predicted_id=choice)
                else:
                    if isinstance(values, list) and 'option_ids' in pred:
                        pred_ids = pred['option_ids']
                        if len(pred_ids) != len(values) or len(set(pred_ids)) != len(pred_ids):
                            raise ValueError('Invalid prediction option IDs')
                        values = dict(zip(pred_ids, values))
                    ps = vector(values, ids)
                    chosen = max(range(len(ps)), key=lambda k: ps[k])
                    if contract is None or contract['hard_projection'] is not None:
                        label = item['label'] if contract is None else contract['hard_projection']['label']
                        row.update(status='distribution', predicted_id=ids[chosen], probabilities=ps,
                                   confidence=ps[chosen], nll=-math.log(max(ps[label], NLL_PROBABILITY_FLOOR)),
                                   brier=sum((p-(k == label))**2 for k,p in enumerate(ps)))
                        if target is not None:
                            row['analytic'] = dict(squared_probability_error=sum((p-q)**2 for p,q in zip(ps,target)),
                                expected_nll=-sum(q*math.log(max(p,NLL_PROBABILITY_FLOOR)) for p,q in zip(ps,target)),
                                expected_brier=1+sum(p*p-2*p*q for p,q in zip(ps,target)))
                    if contract is not None:
                        answer_set = set(contract['answer_set'])
                        row.update(status='distribution', predicted_id=ids[chosen], probabilities=ps,
                                   confidence=ps[chosen],
                                   expected_nll=-sum(q*math.log(max(p, NLL_PROBABILITY_FLOOR)) for p,q in zip(ps,target)),
                                   expected_brier=1+sum(p*p-2*p*q for p,q in zip(ps,target)),
                                   acceptable_set_probability_mass=sum(
                                       probability for option_id, probability in zip(ids, ps)
                                       if option_id in answer_set
                                   ), top_1_in_set=top_1_in_answer_set(ps, ids, answer_set))
                if contract is None or contract['hard_projection'] is not None:
                    row['correct'] = row['predicted_id'] == row['gold_id']
                if contract is not None:
                    if row['status'] == 'native_decision':
                        row['top_1_in_set'] = row['predicted_id'] in set(contract['answer_set'])
                    if contract['hard_projection'] is None:
                        row['correct'] = row['top_1_in_set']
            except (ValueError, TypeError) as exc:
                row.update(status='invalid', error=str(exc))
        rows.append(row)
    return rows


def clusters(rows, key='group_id'):
    result = defaultdict(list)
    for row in rows:
        result[row[key]].append(row)
    return result


def basic(rows):
    if not rows:
        return dict(n=0, accuracy=None, balanced_accuracy=None, macro_f1=None)
    recalls, f1 = [], []
    for label in sorted({r['gold_id'] for r in rows}):
        tp = sum(r['gold_id']==label and r['predicted_id']==label for r in rows)
        fp = sum(r['gold_id']!=label and r['predicted_id']==label for r in rows)
        fn = sum(r['gold_id']==label and r['predicted_id']!=label for r in rows)
        recalls.append(tp/(tp+fn))
        f1.append(2*tp/(2*tp+fp+fn))
    return dict(n=len(rows), accuracy=sum(r['correct'] for r in rows)/len(rows),
                balanced_accuracy=sum(recalls)/len(recalls), macro_f1=sum(f1)/len(f1))


def _summarize_hard(rows):
    result = basic(rows)
    if not rows:
        return result
    groups = list(clusters(rows).values())
    valid = [r for r in rows if r['status']=='distribution']
    result.update(source_groups=len(groups), invalid_or_missing=sum(r['status'] in ('invalid','missing') for r in rows),
                  all_decisions_correct_group_rate=sum(all(r['correct'] for r in g) for g in groups)/len(groups),
                  probability_rows=len(valid), probability_coverage=len(valid)/len(rows))
    for key in ('nll','brier'):
        value = sum(r[key] for r in valid)/len(valid) if valid else None
        result[key] = value if len(valid)==len(rows) else None
        result[key+'_valid_distributions_only'] = value
    result['nll_probability_floor'] = NLL_PROBABILITY_FLOOR
    ranked = sorted(valid, key=lambda r: (-r['confidence'],r['id']))
    result['risk_coverage'] = []
    for fraction in (.25,.5,.75,1):
        if not ranked:
            break
        k = min(len(ranked), max(1,math.ceil(len(rows)*fraction)))
        threshold = ranked[k-1]['confidence']
        accepted = [r for r in ranked if r['confidence']>=threshold]
        result['risk_coverage'].append(dict(requested_coverage=fraction, coverage=len(accepted)/len(rows),
            n=len(accepted), threshold=threshold, error=1-sum(r['correct'] for r in accepted)/len(accepted)))
    result['reliability_bins'] = []
    for b in range(10):
        part = [r for r in valid if min(9,int(r['confidence']*10))==b]
        if part:
            result['reliability_bins'].append(dict(lower=b/10,upper=(b+1)/10,n=len(part),
                mean_confidence=sum(r['confidence'] for r in part)/len(part),
                accuracy=sum(r['correct'] for r in part)/len(part)))
    rng, values = random.Random(217), []
    for _ in range(1000):
        draw = [groups[rng.randrange(len(groups))] for _ in groups]
        values.append(sum(r['correct'] for g in draw for r in g)/sum(map(len,draw)))
    values.sort()
    result['accuracy_cluster_bootstrap_95'] = [values[25],values[974]]
    return result


def summarize_answer_set(rows):
    """Summarize soft answer-set records without coercing native decisions to probabilities."""
    distributions = [row for row in rows if row['status']=='distribution']
    decisions = [row for row in rows if row['status'] in ('distribution', 'native_decision')]
    result = dict(n=len(rows), source_groups=len(clusters(rows)),
                  invalid_or_missing=sum(row['status'] in ('invalid', 'missing') for row in rows),
                  valid_decision_rows=len(decisions),
                  native_decision_rows=sum(row['status']=='native_decision' for row in rows),
                  top_1_in_set=sum(row['top_1_in_set'] for row in rows)/len(rows),
                  top_1_in_set_valid_decisions_only=(
                      sum(row['top_1_in_set'] for row in decisions)/len(decisions) if decisions else None
                  ), probability_rows=len(distributions), probability_coverage=len(distributions)/len(rows))
    for metric, key in (
        ('expected_nll', 'expected_nll'),
        ('brier', 'expected_brier'),
        ('acceptable_set_probability_mass', 'acceptable_set_probability_mass'),
    ):
        value = sum(row[key] for row in distributions)/len(distributions) if distributions else None
        result[metric] = value if len(distributions)==len(rows) else None
        result[metric+'_valid_distributions_only'] = value
    result['nll_probability_floor'] = NLL_PROBABILITY_FLOOR
    return result


def summarize(rows):
    answer_set_rows = [row for row in rows if 'answer_set' in row]
    if not answer_set_rows:
        return _summarize_hard(rows)
    result = _summarize_hard([row for row in rows if 'gold_id' in row])
    result['answer_set_cohort'] = summarize_answer_set(answer_set_rows)
    return result


def balanced_metric(rows):
    hard_rows = [row for row in rows if 'gold_id' in row]
    families = clusters(hard_rows,'family')
    return sum(basic(part)['balanced_accuracy'] for part in families.values())/len(families) if families else None


def paired_comparison(left, right, samples=1000, seed=217):
    """Use identical source-group draws on both systems, stratified by family."""
    unsupported_records = [row['id'] for row in [*left, *right] if 'gold_id' not in row]
    if unsupported_records:
        records = ', '.join(repr(record_id) for record_id in sorted(set(unsupported_records)))
        raise ValueError(
            'Paired comparison requires legacy hard labels or complete singleton hard projections; '
            f'unsupported answer-set records: {records}'
        )
    a,b = indexed(left,'left'),indexed(right,'right')
    if a.keys()!=b.keys() or not a or samples < 40:
        raise ValueError('Need same nonempty gold IDs and at least 40 bootstrap samples')
    for key in a:
        if any(a[key][f]!=b[key][f] for f in ('group_id','family','gold_id')):
            raise ValueError('Paired gold metadata disagree')
    strata = defaultdict(dict)
    for group,rows in clusters(left).items():
        families = {r['family'] for r in rows}
        if len(families)!=1:
            raise ValueError('Source group crosses families')
        strata[next(iter(families))][group] = [r['id'] for r in rows]
    rng,draws = random.Random(seed),[]
    for _ in range(samples):
        ids = []
        for groups in strata.values():
            units = list(groups.values())
            for _ in units:
                ids.extend(units[rng.randrange(len(units))])
        draws.append(balanced_metric([a[k] for k in ids])-balanced_metric([b[k] for k in ids]))
    draws.sort()
    return dict(metric='mean_family_balanced_accuracy', difference=balanced_metric(left)-balanced_metric(right),
        paired_source_group_bootstrap_95=[draws[int(.025*samples)],draws[min(samples-1,int(.975*samples))]],
        source_groups=sum(map(len,strata.values())),samples=samples,seed=seed,
        method='Paired source-group bootstrap stratified by family; represented gold classes per draw')


def evaluate(gold,predictions,comparison=None):
    rows = align(gold,predictions)
    families = {name:summarize(part) for name,part in clusters(rows,'family').items()}
    answer_set_rows = [row for row in rows if 'answer_set' in row]
    hard_families = [result for result in families.values() if result['n']]
    analytic = [dict(id=r['id'],**r['analytic']) for r in rows if 'analytic' in r]
    limitations = ['All gold rows count in accuracy, including missing/invalid/unparsed predictions.',
                   'Native decisions have no probability estimates; one-hot distributions are not fabricated.',
                   'Risk/coverage is descriptive; tied confidence values remain together.',
                   'Analytic event hard-label accuracy measures modal-class agreement, not observed-event accuracy.']
    if answer_set_rows:
        limitations[0] = (
            'All hard-label rows count in accuracy and all answer-set rows count in top-1-in-set, '
            'including missing/invalid/unparsed predictions.'
        )
    result = dict(available_gold=len(gold),scored=len(predictions),evaluated=len(rows),
        coverage=sum(r['status'] not in ('invalid','missing') for r in rows)/len(rows) if rows else 0,
        missing=sum(r['status']=='missing' for r in rows),invalid=sum(r['status']=='invalid' for r in rows),
        family_results=families,mean_family_balanced_accuracy=balanced_metric(rows) if rows else None,
        mean_family_macro_f1=sum(r['macro_f1'] for r in hard_families)/len(hard_families) if hard_families else None,
        analytic_uncertainty={'n':len(analytic),'rows':analytic},errors=[r for r in rows if not r['correct']],
        limitations=limitations)
    if answer_set_rows:
        result['answer_set_cohort'] = summarize_answer_set(answer_set_rows)
        result['limitations'].extend([
            'Multi-answer and unprojected singleton answer-set rows are excluded from hard-label accuracy, '
            'screening, and paired comparisons.',
            'Answer-set probability metrics require native distributions; native decisions contribute only top-1-in-set.'
        ])
    if comparison is not None:
        result['paired_comparison'] = paired_comparison(rows,align(gold,comparison))
    return result


def screening_gate(gold, predictions, policy='distribution', threshold=0.8):
    """Frozen semantic-decision gate; not a benchmark of executed workflow actions."""
    if policy not in ('distribution', 'native') or threshold != 0.8:
        raise ValueError('Screen policy is frozen: distribution p>=0.8 or native parsed decisions')
    require_hard_label_gold(gold, 'Screening gate')
    rows = align(gold, predictions)
    result_by_id = indexed(rows, 'aligned')
    groups = clusters(gold)
    family_counts, family_pairs = defaultdict(int), defaultdict(int)
    issues, pair_correct, missing_automatic, original_correct_automatic = [], 0, 0, 0
    missing_count, evidence_correct, original_automatic = 0, 0, 0
    decisions = []
    required = {'original', 'criterion_reversal', 'evidence_change', 'missing'}
    for group, items in groups.items():
        variants = {}
        for item in items:
            variant = item.get('provenance', {}).get('variant')
            if variant in variants:
                issues.append(f'{group}: duplicate variant {variant}')
            variants[variant] = item
        if set(variants) != required or len(items) != 4 or len({r['family'] for r in items}) != 1:
            issues.append(f'{group}: requires four declared variants in one family')
            continue
        family = items[0]['family']
        family_counts[family] += 1
        paired = all(result_by_id[variants[v]['id']]['correct'] for v in ('original','criterion_reversal'))
        pair_correct += paired
        family_pairs[family] += paired
        evidence_correct += result_by_id[variants['evidence_change']['id']]['correct']
        for variant, item in variants.items():
            row = result_by_id[item['id']]
            semantic_choice = row['predicted_id']
            available = row['status'] == ('distribution' if policy == 'distribution' else 'native_decision')
            automatic = available and semantic_choice != 'insufficient'
            if policy == 'distribution':
                automatic = automatic and row['confidence'] >= threshold
            decisions.append(dict(id=row['id'], variant=variant, choice=semantic_choice,
                                  disposition='automatic_semantic_decision' if automatic else 'review',
                                  correct=row['correct']))
            if variant == 'missing':
                missing_count += 1
                if row['gold_id'] != 'insufficient':
                    issues.append(f'{item["id"]}: missing variant must have insufficient gold')
                missing_automatic += automatic
            elif variant == 'original':
                original_automatic += automatic
                original_correct_automatic += automatic and row['correct']
    expected_families = {'evidence_interpretation','rule_application','candidate_selection'}
    if len(gold)!=96 or len(groups)!=24 or set(family_counts)!=expected_families or any(n!=8 for n in family_counts.values()):
        issues.append('Screen requires 96 rows, 24 groups and eight groups in each declared family')
    complete = all(r['status'] == ('distribution' if policy == 'distribution' else 'native_decision') for r in rows)
    checks = dict(schema=not issues, complete_valid_outputs=complete,
                  original_reversal_pairs=pair_correct>=20,
                  every_family_pairs=set(family_counts)==expected_families and all(family_pairs[f]>=6 for f in expected_families),
                  no_unsupported_missing_decision=missing_count==24 and missing_automatic==0,
                  useful_original_coverage=original_correct_automatic>=12)
    return dict(targets_met=all(checks.values()), structurally_valid=not issues,
                decision='Reviewer decides whether observed failures justify a bounded pilot; numerical near-misses are not automatic vetoes.',
                checks=checks, schema_errors=issues,
                policy=dict(version='screen-policy-v1', mode=policy,
                            threshold=threshold if policy=='distribution' else None,
                            abstain_option='insufficient', scope='semantic decisions, not executed workflow actions'),
                original_reversal_pairs_correct=pair_correct, family_pairs_correct=dict(family_pairs),
                evidence_change_correct=evidence_correct, missing_cases=missing_count,
                unsupported_missing_decisions=missing_automatic,
                original_automatic_decisions=original_automatic,
                original_correct_automatic_decisions=original_correct_automatic,
                accuracy_including_abstentions=basic(rows)['accuracy'], decisions=decisions,
                limitation='Small falsification screen; zero errors does not certify safety/calibration.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('gold','predictions','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--comparison')
    parser.add_argument('--screen-policy', choices=('distribution','native'))
    args=parser.parse_args()
    gold, predictions = read_jsonl(args.gold), read_jsonl(args.predictions)
    if args.screen_policy:
        require_hard_label_gold(gold, 'Screening gate')
    result=evaluate(gold,predictions,read_jsonl(args.comparison) if args.comparison else None)
    if args.screen_policy:
        result['screening_gate'] = screening_gate(gold, predictions, args.screen_policy)
    with Path(args.output).open('x') as destination:
        destination.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('scored','coverage','mean_family_balanced_accuracy')}))
